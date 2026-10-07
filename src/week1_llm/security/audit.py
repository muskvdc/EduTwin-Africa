from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    """Append-only JSONL security audit log.

    Raw user prompts, model outputs, API keys and other sensitive payloads are
    deliberately not written. Callers should provide classifications and safe
    metadata only.
    """

    def __init__(self, path: str | Path = "data/security/audit.jsonl"):
        self.path = Path(path)
        self._lock = threading.Lock()

    @staticmethod
    def hash_value(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def log(
        self,
        event_type: str,
        *,
        request_id: str = "",
        actor_id: str = "",
        severity: str = "info",
        details: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "request_id": request_id,
            "actor_hash": self.hash_value(actor_id) if actor_id else "",
            "severity": severity,
            "details": details or {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
