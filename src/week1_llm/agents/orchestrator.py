"""Coordinates Researcher → Judge feedback loop → Writer."""

from __future__ import annotations

from typing import Any

from .judge import JudgeAgent
from .researcher import ResearcherAgent
from .writer import WriterAgent


class AgentOrchestrator:
    name = "orchestrator"

    def __init__(
        self,
        client: Any,
        context_manager: Any,
        max_iterations: int = 3,
    ):
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        self.client = client
        self.context_manager = context_manager
        self.max_iterations = max_iterations
        self.researcher = ResearcherAgent(client, context_manager)
        self.judge = JudgeAgent(client, context_manager)
        self.writer = WriterAgent(client, context_manager)

    @staticmethod
    def _event(agent: str, status: str, summary: str, attempt: int | None = None):
        event = {"agent": agent, "status": status, "summary": summary}
        if attempt is not None:
            event["attempt"] = attempt
        return event

    @staticmethod
    def _source_links(tool_observations: list[dict[str, Any]] | None) -> list[dict[str, str]]:
        """Extract bounded source metadata so reviewed answers can preserve citations."""
        sources: list[dict[str, str]] = []
        seen: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                url = value.get("url")
                title = value.get("title")
                if isinstance(url, str) and url.startswith(("https://", "http://")):
                    if url not in seen:
                        seen.add(url)
                        sources.append({
                            "title": str(title or url)[:200],
                            "url": url[:500],
                        })
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(tool_observations or [])
        return sources[:5]

    @classmethod
    def _attach_source_links(
        cls,
        research: str,
        tool_observations: list[dict[str, Any]] | None,
    ) -> str:
        sources = cls._source_links(tool_observations)
        if not sources:
            return research

        lines = ["", "", "Source links from the approved web-search tool observations:"]
        lines.extend(f"- {item['title']}: {item['url']}" for item in sources)
        return research + "\n".join(lines)

    def run(
        self,
        request: str,
        base_messages: list[dict[str, str]],
        temperature: float = 0.7,
        tool_observations: list[dict[str, Any]] | None = None,
        tool_trace: list[dict[str, Any]] | None = None,
        adaptive_instruction: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(request, str) or not request.strip():
            raise ValueError("request must be a non-empty string")

        trace = [
            self._event(
                "Orchestrator", "started",
                "Selected the four-agent workflow and prepared the request context.",
            )
        ]
        feedback = ""
        research = ""
        judge_result = None
        attempts = 0

        if tool_trace:
            trace.extend(tool_trace)
        elif tool_observations:
            trace.append(self._event(
                "Tool", "complete",
                "Executed only allowlisted tools; observations were passed as untrusted data.",
            ))

        for attempt in range(1, self.max_iterations + 1):
            attempts = attempt
            research = self.researcher.run(
                request,
                base_messages,
                feedback,
                tool_observations=tool_observations,
                adaptive_instruction=adaptive_instruction,
            )
            research = self._attach_source_links(research, tool_observations)
            trace.append(self._event(
                "Researcher", "complete",
                "Prepared focused findings from the available conversation and retrieved context.",
                attempt,
            ))

            judge_result = self.judge.evaluate(
                request,
                research,
                adaptive_instruction=adaptive_instruction,
            )
            if judge_result.status == "pass":
                trace.append(self._event(
                    "Judge", "approved",
                    judge_result.feedback or "Findings met the review criteria.",
                    attempt,
                ))
                break

            feedback = judge_result.feedback or "Address gaps and unsupported claims."
            trace.append(self._event(
                "Judge", "revision_requested",
                feedback[:360],
                attempt,
            ))

        approved = bool(judge_result and judge_result.status == "pass")
        review_status = "approved" if approved else "review_limit_reached"

        if approved:
            # The Writer is only allowed to produce learner-facing content
            # after the Judge has explicitly approved the findings.
            answer = self.writer.run(
                request=request,
                research=research,
                review_status=review_status,
                judge_feedback=judge_result.feedback if judge_result else "",
                temperature=temperature,
                adaptive_instruction=adaptive_instruction,
            )
            trace.append(self._event(
                "Writer", "complete", "Prepared the final user-facing response."
            ))
            completion_summary = "Workflow finished with Judge approval."
        else:
            # Fail closed: never present an unapproved Writer draft to the
            # learner. Keep diagnostics internally, but show only a safe retry
            # message to the learner.
            answer = (
                "I couldn't complete a fully reviewed lesson response this time. "
                "Your learning progress has been kept. Please try the question "
                "again, or switch to Normal Chat if you need an immediate answer."
            )
            trace.append(self._event(
                "Writer", "skipped",
                "Skipped the final response because the Judge did not approve the findings."
            ))
            completion_summary = (
                "Workflow stopped at the review limit without Judge approval; "
                "no unapproved answer was shown to the learner."
            )

        trace.append(self._event(
            "Orchestrator", "complete", completion_summary
        ))
        return {
            "answer": answer,
            "trace": trace,
            "iterations": attempts,
            "review_status": review_status,
        }
