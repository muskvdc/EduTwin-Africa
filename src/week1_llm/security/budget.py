from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class BudgetExceeded(RuntimeError):
    """Raised when a request would exceed an explicit resource budget."""


@dataclass
class RequestBudget:
    max_model_calls: int = 8
    max_tool_calls: int = 4
    max_estimated_tokens: int = 30000
    model_calls: int = 0
    tool_calls: int = 0
    estimated_tokens: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def consume_model_call(
        self,
        messages: list[dict[str, str]],
        max_completion_tokens: int | None,
    ) -> None:
        if self.model_calls >= self.max_model_calls:
            raise BudgetExceeded(
                "The request reached its model-call safety budget."
            )
        estimated_input = sum(
            max(1, len(str(message.get("content", ""))) // 4)
            for message in messages
        )
        estimated_output = max_completion_tokens or 0
        projected = self.estimated_tokens + estimated_input + estimated_output
        if projected > self.max_estimated_tokens:
            raise BudgetExceeded(
                "The request reached its estimated token safety budget."
            )
        self.model_calls += 1
        self.estimated_tokens = projected
        self.events.append(
            {
                "type": "model_call",
                "call_number": self.model_calls,
                "estimated_tokens": estimated_input + estimated_output,
            }
        )

    def consume_tool_call(self, tool_name: str) -> None:
        if self.tool_calls >= self.max_tool_calls:
            raise BudgetExceeded(
                "The request reached its tool-call safety budget."
            )
        self.tool_calls += 1
        self.events.append(
            {"type": "tool_call", "tool": tool_name, "call_number": self.tool_calls}
        )


class BudgetedClient:
    """Wrap an LLM client so every model call is budget-checked."""

    def __init__(self, client: Any, budget: RequestBudget):
        self._client = client
        self._budget = budget
        self.default_model = getattr(client, "default_model", "")

    def chat(self, messages, temperature=0.7, max_completion_tokens=None, **kwargs):
        self._budget.consume_model_call(messages, max_completion_tokens)
        return self._client.chat(
            messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            **kwargs,
        )
