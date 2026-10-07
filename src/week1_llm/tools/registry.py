from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from week1_llm.security.budget import RequestBudget


RiskLevel = Literal["low", "medium", "high"]


class ToolDenied(RuntimeError):
    pass


class ToolApprovalRequired(RuntimeError):
    pass


class ToolExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    """A tool's public contract plus its trusted Python implementation.

    The model-facing contract is deliberately explicit:
    1. name
    2. description
    3. parameters (JSON Schema)

    The handler and validator stay behind the registry boundary so the model
    never receives arbitrary Python execution capability.
    """

    name: str
    description: str
    handler: Callable[[dict[str, Any]], Any]
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    )
    risk_level: RiskLevel = "low"
    requires_approval: bool = False
    enabled: bool = True
    validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None


@dataclass
class ToolExecutionContext:
    request_id: str
    actor_id: str
    budget: RequestBudget
    allowed_tools: set[str] = field(default_factory=set)
    approved_tools: set[str] = field(default_factory=set)


class ToolRegistry:
    """Explicit tool allowlist with schemas, validation and approval gates."""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.name or not spec.name.strip():
            raise ValueError("Tool name cannot be empty")
        if not spec.description or not spec.description.strip():
            raise ValueError("Tool description cannot be empty")
        if not isinstance(spec.parameters, dict):
            raise ValueError("Tool parameters must be a JSON Schema object")
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the complete educational tool contract for each tool."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
                "risk_level": spec.risk_level,
                "requires_approval": spec.requires_approval,
                "enabled": spec.enabled,
            }
            for spec in self._tools.values()
            if spec.enabled
        ]

    def list_tools_item_names(self) -> set[str]:
        """Return enabled tool names for request-level allowlisting."""
        return {
            spec.name for spec in self._tools.values() if spec.enabled
        }

    def model_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function schemas for future native tool use."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._tools.values()
            if spec.enabled
        ]

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolDenied(f"Unknown tool: {name}") from exc

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> Any:
        spec = self.get(name)
        if not spec.enabled:
            raise ToolDenied(f"Tool is disabled: {name}")
        if context.allowed_tools and name not in context.allowed_tools:
            raise ToolDenied(f"Tool is not allowed for this request: {name}")
        if spec.requires_approval and name not in context.approved_tools:
            raise ToolApprovalRequired(f"Human approval is required for: {name}")
        if not isinstance(arguments, dict):
            raise ToolExecutionError("Tool arguments must be a JSON object")

        try:
            validated = spec.validator(arguments) if spec.validator else dict(arguments)
            context.budget.consume_tool_call(name)
            result = spec.handler(validated)
            json.dumps(result)  # Tool outputs must be serializable before entering model context.
            return result
        except (ToolDenied, ToolApprovalRequired, ToolExecutionError):
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Tool '{name}' failed safely: {type(exc).__name__}") from exc
