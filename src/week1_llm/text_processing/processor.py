from __future__ import annotations

import re
from datetime import date
from typing import Iterable


_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.I)
_HANDLE_RE = re.compile(r"(?<![\w@])@([A-Z0-9_]{1,30})(?![A-Z0-9_])", re.I)
_DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}-\d{1,2}-\d{1,2}|"
    r"(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:\d{2}|\d{4})|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?"
    r")\b",
    re.I,
)
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-])\d{3,4}[ .-]\d{3,4}(?!\w)"
    r"|(?<!\w)\+\d{10,15}(?!\w)"
)
_NUMBER_RE = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)?%?(?![\w])")


def _mask_spans(text: str, spans: Iterable[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            if chars[i] not in "\r\n":
                chars[i] = " "
    return "".join(chars)


def _valid_numeric_date(value: str) -> bool:
    try:
        if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", value):
            year, month, day = map(int, value.split("-"))
        else:
            month, day, year = map(int, value.split("/"))
            if year < 100:
                year += 2000 if year < 70 else 1900
        date(year, month, day)
        return True
    except (ValueError, TypeError):
        return False


class TextProcessor:
    """Text utilities with mutually exclusive extraction categories.

    Regex extraction is heuristic, not a full named-entity recognizer.
    """

    @staticmethod
    def clean_text(text: str, *, normalize_whitespace: bool = True) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        if normalize_whitespace:
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = "\n".join(line.strip() for line in text.splitlines()).strip()
        return text

    @staticmethod
    def extract_emails(text: str) -> list[str]:
        return list(dict.fromkeys(match.group(0) for match in _EMAIL_RE.finditer(text)))

    @staticmethod
    def extract_social_handles(text: str) -> list[str]:
        masked = _mask_spans(text, ((m.start(), m.end()) for m in _EMAIL_RE.finditer(text)))
        return list(dict.fromkeys("@" + m.group(1) for m in _HANDLE_RE.finditer(masked)))

    @staticmethod
    def extract_dates(text: str) -> list[str]:
        found: list[str] = []
        for match in _DATE_RE.finditer(text):
            value = match.group(0)
            if ("/" in value or re.match(r"\d{4}-", value)) and not _valid_numeric_date(value):
                continue
            found.append(value)
        return list(dict.fromkeys(found))

    @staticmethod
    def extract_phone_numbers(text: str) -> list[str]:
        date_spans = [
            (m.start(), m.end()) for m in _DATE_RE.finditer(text)
            if ("/" not in m.group(0) and not re.match(r"\d{4}-", m.group(0))) or _valid_numeric_date(m.group(0))
        ]
        masked = _mask_spans(text, date_spans)
        return list(dict.fromkeys(m.group(0).strip() for m in _PHONE_RE.finditer(masked)))

    @classmethod
    def extract_entities(cls, text: str) -> dict[str, list[str]]:
        emails = cls.extract_emails(text)
        dates = cls.extract_dates(text)
        phones = cls.extract_phone_numbers(text)
        excluded_spans = []
        for pattern in (_EMAIL_RE, _DATE_RE, _PHONE_RE):
            for match in pattern.finditer(text):
                value = match.group(0)
                if pattern is not _DATE_RE or (("/" not in value and not re.match(r"\d{4}-", value)) or _valid_numeric_date(value)):
                    excluded_spans.append((match.start(), match.end()))
        masked = _mask_spans(text, excluded_spans)
        numbers = list(dict.fromkeys(m.group(0) for m in _NUMBER_RE.finditer(masked)))
        return {
            "emails": emails,
            "social_handles": cls.extract_social_handles(text),
            "dates": dates,
            "phone_numbers": phones,
            "numbers": numbers,
        }
