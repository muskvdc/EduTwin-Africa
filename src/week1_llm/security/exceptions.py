from __future__ import annotations

from .guardrails import GuardrailDecision


class SecurityBlocked(RuntimeError):
    """Raised when a security or safety guardrail blocks a request."""

    def __init__(self, decision: GuardrailDecision):
        self.decision = decision
        super().__init__(decision.user_message)
