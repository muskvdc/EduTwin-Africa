from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_GRADES = [f"Grade {grade}" for grade in range(8, 13)]

# These are presentation labels only. Curriculum content remains the source of truth.
SUBJECT_LABELS = {
    "mathematics": "Mathematics",
    "math": "Mathematics",
    "english": "English",
    "natural_science": "Natural Science",
    "natural sciences": "Natural Science",
    "science": "Natural Science",
    "social_science": "Social Science",
    "social sciences": "Social Science",
    "geography": "Geography",
    "history": "History",
}

@dataclass(frozen=True)
class CurriculumSource:
    source: str
    grade: str
    subject: str
    topic: str
    title: str


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).strip()


def _grade_label(raw: str) -> str:
    match = re.search(r"(\d+)", raw)
    return f"Grade {int(match.group(1))}" if match else raw.strip()


def _display_subject(raw: str) -> str:
    key = re.sub(r"[\s-]+", "_", raw.strip().lower())
    return SUBJECT_LABELS.get(key, _clean(raw).title())


def _display_topic(raw: str) -> str:
    return _clean(raw).replace("_", " ").title()


def _parse_source(path: Path) -> CurriculumSource | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    metadata: dict[str, str] = {}
    for line in text.splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        if key in {"grade", "subject", "topic", "title"} and key not in metadata:
            metadata[key] = _clean(value)

    filename = path.stem.lower()
    grade_match = re.search(r"grade[_-]?(\d+)", filename)
    grade = _grade_label(metadata.get("grade", grade_match.group(1) if grade_match else ""))

    subject = metadata.get("subject", "")
    topic = metadata.get("topic", "")
    if not subject or not topic:
        parts = path.stem.split("_")
        if len(parts) >= 3 and not subject:
            subject = parts[1]
        if len(parts) >= 3 and not topic:
            topic = "_".join(parts[2:])

    if not grade or grade not in SUPPORTED_GRADES or not subject or not topic:
        return None

    title = metadata.get("title") or topic
    return CurriculumSource(
        source=path.name,
        grade=grade,
        subject=_display_subject(subject),
        topic=_display_topic(topic),
        title=_clean(title),
    )


def discover_curriculum(knowledge_dir: str | Path) -> list[CurriculumSource]:
    directory = Path(knowledge_dir)
    if not directory.exists():
        return []
    sources = []
    for path in sorted(directory.rglob("*.txt")):
        parsed = _parse_source(path)
        if parsed is not None:
            sources.append(parsed)
    return sources


class CurriculumCatalog:
    """Discovers curriculum choices from the project's .txt knowledge base."""

    def __init__(self, knowledge_dir: str | Path):
        self.knowledge_dir = Path(knowledge_dir)
        self.sources = discover_curriculum(self.knowledge_dir)

    @property
    def grades(self) -> list[str]:
        return list(SUPPORTED_GRADES)

    def subjects_for_grade(self, grade: str) -> list[str]:
        return sorted({item.subject for item in self.sources if item.grade == grade})

    def topics_for(self, grade: str, subject: str) -> list[str]:
        return sorted({
            item.topic
            for item in self.sources
            if item.grade == grade and item.subject == subject
        })

    def sources_for(self, grade: str, subject: str, topic: str | None = None) -> list[str]:
        return [
            item.source
            for item in self.sources
            if item.grade == grade
            and item.subject == subject
            and (topic is None or item.topic == topic)
        ]

    def has_curriculum(self, grade: str, subject: str, topic: str | None = None) -> bool:
        return bool(self.sources_for(grade, subject, topic))

    def summary(self, grade: str, subject: str, topic: str | None = None) -> str:
        sources = self.sources_for(grade, subject, topic)
        if not sources:
            return "No matching curriculum source is loaded yet."
        return f"{len(sources)} curriculum source(s) loaded."
