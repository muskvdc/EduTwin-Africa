"""Shared per-request agent/tool runtime used by all EduTwin execution paths."""

from __future__ import annotations

from typing import Any

from week1_llm.tools import (
    ReActResult,
    ReActToolAgent,
    ToolExecutionContext,
    build_default_tool_registry,
)


class AgentRuntime:
    """Build and execute the shared bounded tool/ReAct layer.

    The runtime owns request-scoped tool registration and ReAct execution so
    Normal Chat and Multi-Agent workflows use the same allowlist, validation,
    security budget, curriculum source filter, and web-search capability.
    """

    def __init__(self, *, rag_index: Any, memory: Any, context_manager: Any):
        self.rag_index = rag_index
        self.memory = memory
        self.context_manager = context_manager

    def request_context(
        self,
        *,
        request_id: str,
        actor_id: str,
        budget: Any,
        source_filter: list[str] | tuple[str, ...] | None,
    ) -> tuple[Any, ToolExecutionContext]:
        registry = build_default_tool_registry(
            self.rag_index,
            self.memory,
            knowledge_source_filter=source_filter,
        )
        context = ToolExecutionContext(
            request_id=request_id,
            actor_id=actor_id,
            budget=budget,
            allowed_tools=set(registry.list_tools_item_names()),
        )
        return registry, context

    def react(
        self,
        *,
        client: Any,
        user_input: str,
        request_id: str,
        actor_id: str,
        budget: Any,
        source_filter: list[str] | tuple[str, ...] | None,
        max_steps: int,
    ) -> ReActResult | None:
        registry, tool_context = self.request_context(
            request_id=request_id,
            actor_id=actor_id,
            budget=budget,
            source_filter=source_filter,
        )
        if not ReActToolAgent.should_attempt(user_input):
            return None

        agent = ReActToolAgent(
            client=client,
            context_manager=self.context_manager,
            registry=registry,
            max_steps=max_steps,
        )
        return agent.run(user_input, tool_context)

    @staticmethod
    def observations(result: ReActResult | None) -> list[dict[str, Any]]:
        if not result:
            return []
        observations = list(result.observations)
        if result.answer:
            observations.append({"react_conclusion": result.answer})
        return observations

    @staticmethod
    def trace(result: ReActResult | None) -> list[dict[str, Any]]:
        return list(result.trace) if result else []
