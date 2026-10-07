from __future__ import annotations

import os
import uuid
from typing import Any

from .audit import AuditLogger
from .budget import BudgetedClient, RequestBudget
from .exceptions import SecurityBlocked
from .guardrails import GuardrailDecision, InputGuardrails, OutputGuardrails
from .rate_limit import RateLimiter


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class SecurityManager:
    """Coordinates input/output guardrails, rate limiting, budgets and audit."""

    def __init__(
        self,
        audit_logger: AuditLogger | None = None,
        input_guardrails: InputGuardrails | None = None,
        output_guardrails: OutputGuardrails | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        self.audit = audit_logger or AuditLogger(
            os.getenv("SECURITY_AUDIT_LOG", "data/security/audit.jsonl")
        )
        self.input_guardrails = input_guardrails or InputGuardrails(
            max_chars=_int_env("SECURITY_MAX_INPUT_CHARS", 12000),
            injection_block_score=float(os.getenv("SECURITY_INJECTION_BLOCK_SCORE", "0.70")),
        )
        self.output_guardrails = output_guardrails or OutputGuardrails(
            max_chars=_int_env("SECURITY_MAX_OUTPUT_CHARS", 24000)
        )
        self.rate_limiter = rate_limiter or RateLimiter(
            max_requests=_int_env("SECURITY_RATE_LIMIT_REQUESTS", 30),
            window_seconds=_int_env("SECURITY_RATE_LIMIT_WINDOW_SECONDS", 60),
        )

    def new_request_id(self) -> str:
        return uuid.uuid4().hex

    def check_rate_limit(self, actor_id: str, request_id: str) -> None:
        decision = self.rate_limiter.check(actor_id)
        if not decision.allowed:
            self.audit.log(
                "rate_limit_block",
                request_id=request_id,
                actor_id=actor_id,
                severity="warning",
                details={"retry_after_seconds": decision.retry_after_seconds},
            )
            raise SecurityBlocked(
                GuardrailDecision(
                    False,
                    "Too many requests in a short period. Please wait and try again.",
                    ("rate_limit",),
                    0.8,
                    {"retry_after_seconds": decision.retry_after_seconds},
                )
            )

    def check_input(self, text: str, request_id: str, actor_id: str):
        decision = self.input_guardrails.check(text)
        if decision.reasons or not decision.allowed:
            self.audit.log(
                "input_guardrail",
                request_id=request_id,
                actor_id=actor_id,
                severity="warning" if not decision.allowed else "info",
                details={
                    "allowed": decision.allowed,
                    "reasons": decision.reasons,
                    "risk_score": decision.risk_score,
                    **(decision.metadata or {}),
                },
            )
        if not decision.allowed:
            raise SecurityBlocked(decision)
        return decision

    def check_output(
        self,
        output: str,
        *,
        protected_prompt: str,
        request_id: str,
        actor_id: str,
    ) -> str:
        decision = self.output_guardrails.check(output, protected_prompt)
        if not decision.allowed:
            self.audit.log(
                "output_guardrail_block",
                request_id=request_id,
                actor_id=actor_id,
                severity="critical",
                details={
                    "reasons": decision.reasons,
                    "risk_score": decision.risk_score,
                    **(decision.metadata or {}),
                },
            )
            return decision.user_message

        return output

    def new_budget(self, *, multi_agent: bool) -> RequestBudget:
        # Normal requests usually need one model call. Tool-enabled requests
        # may need one bounded ReAct decision plus one final answer, so keep a
        # small three-call ceiling rather than forcing tool use off.
        default_calls = 10 if multi_agent else 3
        max_calls = _int_env("SECURITY_MAX_MODEL_CALLS", default_calls)
        max_tokens = _int_env("SECURITY_MAX_ESTIMATED_TOKENS", 30000)
        return RequestBudget(
            max_model_calls=max_calls,
            max_tool_calls=_int_env("SECURITY_MAX_TOOL_CALLS", 4),
            max_estimated_tokens=max_tokens,
        )

    def guarded_client(self, client: Any, budget: RequestBudget) -> BudgetedClient:
        return BudgetedClient(client, budget)
