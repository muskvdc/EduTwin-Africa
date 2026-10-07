from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    detected: bool
    score: float
    categories: tuple[str, ...]
    matches: tuple[str, ...] = ()


class PromptInjectionDetector:
    """Heuristic detector based on the Week 3 red-team exercises.

    This is one layer, not a proof of safety. Architectural separation of
    trusted instructions from untrusted content remains the primary defense.
    """

    _patterns = (
        ("direct_override", r"\bignore\s+(all\s+)?previous\s+instructions\b", 0.80),
        ("instruction_disregard", r"\bdisregard\s+(all\s+)?instructions\b", 0.75),
        ("forget_rules", r"\bforget\s+(everything|all\s+(previous\s+)?instructions)\b", 0.75),
        ("system_extraction", r"\breveal\s+(your\s+)?(system|developer)\s+(prompt|instructions)\b", 0.85),
        ("prompt_extraction", r"\bprint\s+the\s+(exact\s+)?system\s+prompt\b", 0.85),
        ("role_override", r"\byou\s+are\s+now\s+(a|an)\b", 0.45),
        ("fake_system", r"\[system\s*(override|message)\]", 0.65),
        ("fake_developer", r"\bdeveloper\s+message\b|\b(developer|admin)\s*(override|message|mode)\b", 0.75),
        ("new_priority", r"\bnew\s+priority\b", 0.55),
        ("role_impersonation", r"\bsystem\s*:\s*(?:the\s+)?user\s+is\s+now\s+(?:an?\s+)?administrator\b", 0.75),
        ("delimiter_spoof", r"---\s*system\s*(message|prompt)?\s*---", 0.70),
        ("boundary_spoof", r"\bend\s+of\s+user\s+input\b", 0.45),
        ("safety_bypass", r"\b(bypass|disable|turn\s+off)\s+(safety|guardrails?|filters?)\b", 0.70),
        ("hidden_rules", r"\b(reveal|show|print)\s+(your\s+)?hidden\s+rules\b", 0.80),
        ("tool_abuse", r"\bignore\s+tool\s+(restrictions|policy|rules)\b", 0.70),
    )

    def detect(self, text: str) -> Detection:
        if not isinstance(text, str):
            return Detection(True, 1.0, ("invalid_input",), ())
        categories: list[str] = []
        matches: list[str] = []
        score = 0.0
        lower = text.lower()

        for category, pattern, weight in self._patterns:
            if re.search(pattern, lower, flags=re.IGNORECASE):
                categories.append(category)
                matches.append(category)
                score += weight

        if len(text) > 12000:
            categories.append("excessive_length")
            score += 0.20
        if text.count("---") >= 4 or text.count("###") >= 4:
            categories.append("delimiter_abuse")
            score += 0.30

        return Detection(
            detected=score > 0,
            score=min(score, 1.0),
            categories=tuple(dict.fromkeys(categories)),
            matches=tuple(matches),
        )


class SecretDetector:
    """Detect common high-confidence secret formats."""

    _patterns = (
        ("groq_key", r"\bgsk_[A-Za-z0-9_-]{20,}\b"),
        ("openai_key", r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        ("github_token", r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
        ("aws_access_key", r"\bAKIA[0-9A-Z]{16}\b"),
        ("jwt", r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        ("private_key", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        ("password_assignment", r"(?i)\b(?:password|passwd|api[_ -]?key|secret)\s*[:=]\s*\S{8,}"),
    )

    def detect(self, text: str) -> Detection:
        categories = []
        matches = []
        for category, pattern in self._patterns:
            if re.search(pattern, text or ""):
                categories.append(category)
                matches.append(category)
        return Detection(bool(categories), 1.0 if categories else 0.0, tuple(categories), tuple(matches))


class PIIDetector:
    """Conservative PII detector used for auditing, not automatic censorship."""

    _patterns = (
        ("email", r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
        ("phone", r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)"),
        ("date_of_birth", r"\b(?:\d{2}[/-]\d{2}[/-]\d{4}|\d{4}-\d{2}-\d{2})\b"),
        ("credit_card_like", r"\b(?:\d[ -]*?){13,19}\b"),
        ("us_ssn_like", r"\b\d{3}-\d{2}-\d{4}\b"),
        ("street_address", r"\b\d{1,5}\s+[A-Za-z0-9][A-Za-z0-9 .,'-]{2,40}\s+(?:Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b"),
    )

    def detect(self, text: str) -> Detection:
        categories = []
        matches = []
        for category, pattern in self._patterns:
            if re.search(pattern, text or "", flags=re.IGNORECASE):
                categories.append(category)
                matches.append(category)
        return Detection(bool(categories), 1.0 if categories else 0.0, tuple(categories), tuple(matches))

    @staticmethod
    def redact_for_logging(text: str) -> str:
        """Return a safe approximation for diagnostics; never used as model input."""
        redacted = re.sub(
            PIIDetector._patterns[0][1], "[REDACTED_EMAIL]", text or "", flags=re.IGNORECASE
        )
        redacted = re.sub(
            PIIDetector._patterns[1][1], "[REDACTED_PHONE]", redacted
        )
        return redacted
