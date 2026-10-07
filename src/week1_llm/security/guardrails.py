from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .detectors import PIIDetector, PromptInjectionDetector, SecretDetector


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    user_message: str
    reasons: tuple[str, ...] = ()
    risk_score: float = 0.0
    metadata: dict[str, Any] | None = None


class InputGuardrails:
    def __init__(
        self,
        max_chars: int = 12000,
        injection_block_score: float = 0.70,
    ):
        self.max_chars = max_chars
        self.injection_block_score = injection_block_score
        self.injection = PromptInjectionDetector()
        self.secrets = SecretDetector()
        self.pii = PIIDetector()

    def check(self, text: str) -> GuardrailDecision:
        if not isinstance(text, str) or not text.strip():
            return GuardrailDecision(False, "Please enter a non-empty message.", ("empty_input",), 1.0)

        if len(text) > self.max_chars:
            return GuardrailDecision(
                False,
                f"That message is too large. Please keep it under {self.max_chars} characters.",
                ("input_too_large",),
                0.8,
            )

        secret = self.secrets.detect(text)
        if secret.detected:
            return GuardrailDecision(
                False,
                "I can't process messages containing what looks like a credential or secret. Remove the secret and try again.",
                ("secret_detected",),
                1.0,
                {"secret_categories": secret.categories},
            )

        injection = self.injection.detect(text)
        pii = self.pii.detect(text)

        if injection.score >= self.injection_block_score:
            return GuardrailDecision(
                False,
                "I can't process that request because it triggered an AI security check. Please rephrase the request without attempts to override system instructions.",
                ("prompt_injection",) + injection.categories,
                injection.score,
                {"injection_categories": injection.categories},
            )

        # PII is audited rather than automatically blocked. Legitimate users may
        # need to discuss their own personal information.
        reasons = tuple(("pii_present",) if pii.detected else ())
        return GuardrailDecision(
            True,
            "",
            reasons,
            injection.score,
            {
                "injection_categories": injection.categories,
                "pii_categories": pii.categories,
            },
        )


class OutputGuardrails:
    def __init__(self, max_chars: int = 24000):
        self.max_chars = max_chars
        self.secrets = SecretDetector()

    @staticmethod
    def _normalized_tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def _possible_prompt_leak(self, output: str, protected_prompt: str) -> bool:
        if not protected_prompt or not output:
            return False
        source = self._normalized_tokens(protected_prompt)
        target = self._normalized_tokens(output)
        if len(source) < 8 or len(target) < 8:
            return False
        target_set = set(zip(*(target[i:] for i in range(8))))
        for i in range(len(source) - 7):
            phrase = tuple(source[i:i + 8])
            if phrase in target_set:
                return True
        return False

    def check(self, output: str, protected_prompt: str = "") -> GuardrailDecision:
        if not isinstance(output, str):
            return GuardrailDecision(False, "The assistant returned an invalid response.", ("invalid_output",), 1.0)
        if len(output) > self.max_chars:
            return GuardrailDecision(
                False,
                "The assistant response exceeded the safety output limit.",
                ("output_too_large",),
                0.8,
            )

        secret = self.secrets.detect(output)
        if secret.detected:
            return GuardrailDecision(
                False,
                "The response was blocked because it appeared to contain a credential or secret.",
                ("secret_leak",),
                1.0,
                {"secret_categories": secret.categories},
            )

        if self._possible_prompt_leak(output, protected_prompt):
            return GuardrailDecision(
                False,
                "The response was blocked because it appeared to reproduce protected system instructions.",
                ("system_prompt_leak",),
                0.95,
            )

        return GuardrailDecision(True, "", (), 0.0, {})
