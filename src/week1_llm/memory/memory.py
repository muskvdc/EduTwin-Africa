from __future__ import annotations

import json
from pathlib import Path


class Memory:
    """Simple local profile/memory foundation for personalization.

    Persistent storage remains the default for the original project/API.
    Streamlit learner sessions can opt into in-memory mode so one learner's
    profile can never be written to or read from a shared local file.
    """

    def __init__(
        self,
        path: str | Path = "data/memory.json",
        *,
        persist: bool = True,
    ):
        self.path = Path(path)
        self.persist = persist
        self.data: dict[str, str] = {}
        if self.persist:
            self.load()

    def load(self) -> None:
        if not self.persist or not self.path.exists():
            return
        self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def set(self, key: str, value: str) -> None:
        self.data[key] = value
        self.save()

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.data.get(key, default)

    def all(self) -> dict[str, str]:
        return dict(self.data)

    def save(self) -> None:
        if not self.persist:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, indent=2),
            encoding="utf-8",
        )

    def as_system_context(self) -> str:
        if not self.data:
            return ""
        lines = ["Known user context:"]
        lines.extend(f"- {key}: {value}" for key, value in self.data.items())
        return "\n".join(lines)
