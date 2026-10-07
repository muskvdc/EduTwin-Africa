"""Judge role: return a conservative, structured review decision."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class JudgeFeedback:
    status: Literal["pass", "revise"]
    feedback: str

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "feedback": self.feedback}


class JudgeAgent:
    name = "judge"

    SYSTEM_PROMPT = (
        "You are the Judge in a four-agent assistant workflow. Evaluate the "
        "Researcher's findings against the original request for relevance, "
        "coverage, internal consistency, and support from supplied context. "
        "An answer may pass when it clearly identifies missing information "
        "instead of guessing. Do not require unsupported details. This workflow "
        "may be used by an adaptive tutor: for broad teaching requests such as "
        "'teach me algebra from the beginning', the Researcher does NOT need to "
        "cover an entire subject in one response. A focused, accurate first "
        "lesson or first concept is acceptable when it is appropriate for the "
        "learner's level and gives the tutor a sensible next step. Do not reject "
        "a focused lesson merely because it does not exhaustively cover every "
        "subtopic. Prefer revise only for factual errors, serious irrelevance, "
        "unsupported claims, unsafe content, or a missing core requirement of "
        "the specific request. Return only JSON with exactly these fields: "
        '{"status":"pass" or "revise","feedback":"brief actionable reason"}. '
        "Use revise when an important problem remains."
    )

    def __init__(self, client: Any, context_manager: Any):
        self.client = client
        self.context_manager = context_manager

    @staticmethod
    def parse(raw: str) -> JudgeFeedback:
        text = (raw or "").strip()
        candidate = text
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
            candidate = re.sub(r"\s*```$", "", candidate)
        if "{" in candidate and "}" in candidate:
            candidate = candidate[candidate.find("{"):candidate.rfind("}") + 1]
        try:
            data = json.loads(candidate)
            status = data.get("status")
            feedback = data.get("feedback", "")
            if status in {"pass", "revise", "fail"} and isinstance(feedback, str):
                normalized = "revise" if status == "fail" else status
                return JudgeFeedback(normalized, feedback.strip())
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

        # Conservative fallback: malformed or ambiguous reviews are not approval.
        return JudgeFeedback(
            "revise",
            "Judge output was not valid structured JSON; review was not approved. "
            + text[:400],
        )

    def evaluate(
        self,
        request: str,
        research: str,
        adaptive_instruction: str | None = None,
    ) -> JudgeFeedback:
        # Reserve room for the original request and Judge instructions; cap only
        # the research payload if it is unusually large.
        prefix = f"Original user request:\n{request}\n\nResearch findings:\n"
        counter = self.context_manager.token_counter
        fixed = self.context_manager.count_messages([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prefix},
        ])
        available = max(32, self.context_manager.input_budget - fixed - 12)
        if counter.count(research) > available:
            research = _truncate_to_tokens(research, available, counter)
        judge_system = self.SYSTEM_PROMPT
        if adaptive_instruction and adaptive_instruction.strip():
            judge_system += (
                "\n\nTrusted adaptive tutoring policy for this turn:\n"
                + adaptive_instruction.strip()
            )
        messages = [
            {"role": "system", "content": judge_system},
            {"role": "user", "content": prefix + research
             + "\n\nReturn the required JSON review."},
        ]
        messages = self.context_manager.fit_messages(messages)
        raw = self.client.chat(
            messages,
            temperature=0,
            max_completion_tokens=min(256, max(1, self.context_manager.reserved_output_tokens)),
        )
        return self.parse(raw)


def _truncate_to_tokens(text: str, limit: int, counter: Any) -> str:
    if limit <= 0:
        return ""
    suffix = "\n[Research excerpt shortened to fit the context budget.]"
    if counter.count(text) <= limit:
        return text
    low, high = 0, len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid].rstrip() + suffix
        if counter.count(candidate) <= limit:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best or text[:max(1, len(text) // 8)]
