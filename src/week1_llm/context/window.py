from __future__ import annotations

import re
from typing import Protocol


class ContextBudgetError(ValueError):
    """Raised when protected instructions and the latest user request cannot fit."""


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class TiktokenTokenCounter:
    """Counts text with tiktoken's model-associated encoding.

    Chat-message framing overhead is added separately by ContextWindowManager.
    Provider-side exact usage may differ because chat templates are model-specific.
    """

    def __init__(self, model: str = "openai/gpt-oss-120b"):
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "Model token accounting requires tiktoken. Run: python -m pip install -r requirements.txt"
            ) from exc

        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # gpt-oss and unknown OpenAI-compatible model aliases may not have
            # a registered name. Prefer the newer OpenAI vocabulary when present.
            encoding_names = ("o200k_harmony", "o200k_base", "cl100k_base") if "gpt-oss" in model.lower() else ("o200k_base", "cl100k_base")
            last_error = None
            for encoding_name in encoding_names:
                try:
                    self.encoding = tiktoken.get_encoding(encoding_name)
                    break
                except ValueError as exc:
                    last_error = exc
            else:
                raise RuntimeError("No compatible tiktoken encoding is installed.") from last_error
        self.model = model

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))


class RegexTokenCounter:
    """Deterministic fallback for tests/demos only; not model-accurate."""

    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def count(self, text: str) -> int:
        return len(self._pattern.findall(text))


class ContextWindowManager:
    """Selects complete recent turns within a token budget.

    System messages and the latest user message are protected. Older history is
    removed from oldest to newest as needed. If protected content alone cannot
    fit, a ContextBudgetError is raised rather than silently deleting instructions.
    """

    MESSAGE_OVERHEAD_TOKENS = 4
    REPLY_PRIMER_TOKENS = 2
    ALLOWED_ROLES = {"system", "user", "assistant"}

    def __init__(
        self,
        token_counter: TokenCounter,
        max_context_tokens: int = 8192,
        reserved_output_tokens: int = 1024,
    ):
        if max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        if reserved_output_tokens < 0 or reserved_output_tokens >= max_context_tokens:
            raise ValueError("reserved_output_tokens must be >= 0 and smaller than max_context_tokens")
        self.token_counter = token_counter
        self.max_context_tokens = max_context_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.input_budget = max_context_tokens - reserved_output_tokens

    def count_message(self, message: dict[str, str]) -> int:
        return self.MESSAGE_OVERHEAD_TOKENS + self.token_counter.count(message["role"]) + self.token_counter.count(message["content"])

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        return self.REPLY_PRIMER_TOKENS + sum(self.count_message(message) for message in messages)

    @staticmethod
    def _validate(messages: list[dict[str, str]]) -> None:
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError(f"Message {index} must contain role and content")
            if message["role"] not in ContextWindowManager.ALLOWED_ROLES:
                raise ValueError(f"Unsupported message role: {message['role']}")
            if not isinstance(message["content"], str):
                raise TypeError(f"Message {index} content must be a string")

    def fit_messages(
        self,
        messages: list[dict[str, str]],
        *,
        token_budget: int | None = None,
        optional_system_indices: set[int] | None = None,
    ) -> list[dict[str, str]]:
        self._validate(messages)
        budget = self.input_budget if token_budget is None else token_budget
        if budget < 1:
            raise ValueError("token_budget must be positive")

        optional_indices = set(optional_system_indices or ())
        for index in optional_indices:
            if index < 0 or index >= len(messages) or messages[index]["role"] != "system":
                raise ValueError("optional_system_indices must point to system messages")

        indexed = list(enumerate(messages))
        required_systems = [
            (i, m) for i, m in indexed
            if m["role"] == "system" and i not in optional_indices
        ]
        optional_systems = [
            (i, m) for i, m in indexed
            if m["role"] == "system" and i in optional_indices
        ]
        non_system = [(i, m) for i, m in indexed if m["role"] != "system"]

        latest_user_position = next(
            (i for i in range(len(non_system) - 1, -1, -1) if non_system[i][1]["role"] == "user"),
            None,
        )
        if latest_user_position is None:
            raise ValueError("At least one user message is required")

        latest_user = non_system[latest_user_position]
        selected: list[tuple[int, dict[str, str]]] = required_systems + [latest_user]
        used = self.count_messages([message for _, message in selected])
        if used > budget:
            raise ContextBudgetError(
                f"Protected system instructions and latest user message need about {used} tokens; budget is {budget}."
            )

        # Retrieved reference context is useful but not an instruction. Include
        # it when it fits; never sacrifice core system instructions or the
        # current user request to make room for it.
        for item in optional_systems:
            cost = self.count_message(item[1])
            if used + cost <= budget:
                selected.append(item)
                used += cost

        history = non_system[:latest_user_position]
        turns: list[list[tuple[int, dict[str, str]]]] = []
        for item in history:
            if item[1]["role"] == "user" or not turns:
                turns.append([item])
            else:
                turns[-1].append(item)

        kept: list[list[tuple[int, dict[str, str]]]] = []
        for turn in reversed(turns):
            cost = sum(self.count_message(message) for _, message in turn)
            if used + cost > budget:
                break
            kept.append(turn)
            used += cost

        selected.extend(item for turn in kept for item in turn)
        selected.sort(key=lambda item: item[0])
        return [{"role": message["role"], "content": message["content"]} for _, message in selected]
