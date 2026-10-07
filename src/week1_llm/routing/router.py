"""State-aware request routing for EduTwin Africa."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class RouteDecision:
    """Deterministic routing decision made before model/tool execution."""

    route: str
    reason: str
    activity: tuple[str, ...]

    @property
    def is_multi_agent(self) -> bool:
        return self.route == "multi_agent"


class RequestRouter:
    """Route learner requests without adding an extra LLM call.

    Routing is intentionally conservative:
    - active adaptive checks have priority so a learner answer is not
      misclassified as a generic tool/research request;
    - current/recent requests can remain single-agent and use ReAct + web;
    - genuinely complex research/verification/comparison can escalate to
      the full Researcher → Judge → Writer workflow.
    """

    _GREETING_PATTERNS = (
        r"^hi$",
        r"^hello$",
        r"^hey$",
        r"^hiya$",
        r"^good morning$",
        r"^good afternoon$",
        r"^good evening$",
        r"^how are you(?: doing)?[?!.]*$",
        r"^how's it going[?!.]*$",
        r"^what's up[?!.]*$",
        r"^what is up[?!.]*$",
        r"^whats up[?!.]*$",
        r"^how's everything[?!.]*$",
        r"^how is everything[?!.]*$",
        r"^how have you been[?!.]*$",
        r"^yo[?!.]*$",
        r"^hey[?!.]*$",
        r"^hiya[?!.]*$",
        r"^sup[?!.]*$",
        r"^howdy[?!.]*$",
        r"^morning[?!.]*$",
        r"^afternoon[?!.]*$",
        r"^evening[?!.]*$",
        r"^good morning[?!.]*$",
        r"^good afternoon[?!.]*$",
        r"^good evening[?!.]*$",
    )


    _ADAPTIVE_CONTROL_PATTERNS = (
        r"^teach me .*",
        r"^teach .*",
        r"^start over[?!.]*$",
        r"^restart[?!.]*$",
        r"^start again[?!.]*$",
        r"^repeat that[?!.]*$",
        r"^explain that again[?!.]*$",
        r"^can you explain that again[?!.]*$",
        r"^skip this[?!.]*$",
    )

    _MULTI_PATTERNS = (
        r"\bresearch\b",
        r"\bdeep dive\b",
        r"\binvestigate\b",
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bcompare and contrast\b",
        r"\bmultiple sources\b",
        r"\bseveral sources\b",
        r"\bfind evidence\b",
        r"\bevaluate (?:the )?sources?\b",
        r"\bconflicting (?:information|sources|claims)\b",
        r"\bverify (?:these|the|this) (?:claims?|facts?|sources?)\b",
        r"\bfact[- ]check\b",
        r"\banalyze (?:the )?(?:evidence|sources|claims)\b",
        r"\bpros and cons\b",
        r"\badvantages and disadvantages\b",
        r"\bmake a case for and against\b",
        r"\bdebate\b",
    )

    _WEB_PATTERNS = (
        r"\blatest\b",
        r"\bmost recent\b",
        r"\brecent news\b",
        r"\bcurrent news\b",
        r"\bnews about\b",
        r"\bcurrent information\b",
        r"\bas of today\b",
        r"\btoday'?s\b",
        r"\bthis week\b",
        r"\bcurrently\b",
        r"\bverify online\b",
        r"\bfact check\b",
        r"\bsearch the web\b",
        r"\bweb search\b",
        r"\bsearch online\b",
        r"\blook (?:it )?up online\b",
        r"\blook online\b",
    )

    @staticmethod
    def _matches(text: str, patterns: tuple[str, ...]) -> bool:
        lowered = text.lower()
        return any(re.search(pattern, lowered) for pattern in patterns)

    @staticmethod
    def _adaptive_state(state: dict[str, Any] | None) -> dict[str, Any]:
        return state if isinstance(state, dict) else {}

    @staticmethod
    def _learner_text(user_input: str) -> str:
        """Extract the learner message from the UI request envelope."""
        text = user_input.strip()
        marker = "\nLearner question:\n"
        if marker in text:
            candidate = text.rsplit(marker, 1)[1].strip()
            if candidate:
                return candidate
        return text

    def route(
        self,
        user_input: str,
        *,
        adaptive_state: dict[str, Any] | None = None,
    ) -> RouteDecision:
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must be a non-empty string")

        state = self._adaptive_state(adaptive_state)
        learner_text = self._learner_text(user_input)

        # Greetings are conversational turns, not learning requests. When a
        # check is active, a greeting is also not an answer to that check: keep
        # the learning state intact and handle the greeting without advancing
        # mastery or generating a hint.
        if self._matches(learner_text, self._GREETING_PATTERNS):
            if bool(state.get("pending_check")):
                return RouteDecision(
                    route="single_agent",
                    reason="adaptive_check_greeting",
                    activity=("Continuing your learning check",),
                )
            return RouteDecision(
                route="single_agent",
                reason="casual_greeting",
                activity=(),
            )

        # Otherwise the learner is answering the tutor's latest check. Explicit
        # tutoring/control requests are not answers and must not trigger a hint
        # or change the adaptive state.
        if bool(state.get("pending_check")):
            if self._matches(learner_text, self._ADAPTIVE_CONTROL_PATTERNS):
                return RouteDecision(
                    route="single_agent",
                    reason="active_adaptive_control",
                    activity=(),
                )
            return RouteDecision(
                route="single_agent",
                reason="active_adaptive_check",
                activity=("Continuing your learning check", "Preparing your explanation"),
            )

        if self._matches(learner_text, self._MULTI_PATTERNS):
            return RouteDecision(
                route="multi_agent",
                reason="complex_research_or_verification",
                activity=("Researching", "Checking", "Preparing your explanation"),
            )

        if self._matches(learner_text, self._WEB_PATTERNS):
            return RouteDecision(
                route="single_agent",
                reason="time_sensitive_or_web_request",
                activity=("Checking current information", "Preparing your explanation"),
            )

        return RouteDecision(
            route="single_agent",
            reason="standard_tutoring_request",
            activity=("Preparing your explanation",),
        )
