from __future__ import annotations

import re
from typing import Any

from .registry import ToolExecutionContext, ToolRegistry


class SafeToolRouter:
    """Small deterministic tool router for the Week 3 project.

    It deliberately uses explicit rules rather than giving the model arbitrary
    function execution. The registry is designed so a future provider-native
    tool-calling layer can replace this router without bypassing the same
    security controls.
    """

    _math_hint = re.compile(r"\b(?:calculate|compute|what is)\b.*[\d][\d\s.+\-*/()%]+", re.I)

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run_for_query(self, query: str, context: ToolExecutionContext) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        lowered = query.lower()

        if self._math_hint.search(query) or (
            any(op in query for op in ("+", "-", "*", "/"))
            and any(ch.isdigit() for ch in query)
        ):
            match = re.search(r"[\d][\d\s.+\-*/()%]*", query)
            if match:
                try:
                    result = self.registry.execute(
                        "calculator",
                        {"expression": match.group(0).strip()},
                        context,
                    )
                    observations.append({"tool": "calculator", "result": result})
                except Exception as exc:
                    observations.append({"tool": "calculator", "error": type(exc).__name__})

        elif any(word in lowered for word in ("current time", "what time", "today's date", "what day is it")):
            try:
                result = self.registry.execute("current_time", {}, context)
                observations.append({"tool": "current_time", "result": result})
            except Exception as exc:
                observations.append({"tool": "current_time", "error": type(exc).__name__})

        return observations
