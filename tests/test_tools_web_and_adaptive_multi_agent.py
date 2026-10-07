
import json
from types import SimpleNamespace

from week1_llm.agents.orchestrator import AgentOrchestrator
from week1_llm.context.window import ContextWindowManager, RegexTokenCounter
from week1_llm.pipeline import AssistantPipeline
from week1_llm.security.budget import RequestBudget
from week1_llm.tools import ReActToolAgent, ToolExecutionContext, build_default_tool_registry
import week1_llm.tools.builtins as builtins


def manager():
    return ContextWindowManager(
        RegexTokenCounter(), max_context_tokens=1800, reserved_output_tokens=200
    )


def test_web_search_is_exposed_when_configured(monkeypatch):
    monkeypatch.setattr(
        builtins,
        "config",
        SimpleNamespace(
            tavily_api_key="test-key",
            web_search_timeout_seconds=10,
            web_search_max_results=5,
        ),
    )
    registry = build_default_tool_registry()
    assert "web_search" in registry.list_tools_item_names()


def test_web_search_returns_compact_untrusted_results(monkeypatch):
    monkeypatch.setattr(
        builtins,
        "config",
        SimpleNamespace(
            tavily_api_key="test-key",
            web_search_timeout_seconds=10,
            web_search_max_results=5,
        ),
    )

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {
                        "title": "Example",
                        "url": "https://example.com",
                        "content": "Fresh result text.",
                    }
                ]
            }

    monkeypatch.setattr(builtins.requests, "post", lambda *args, **kwargs: FakeResponse())

    registry = build_default_tool_registry()
    ctx = ToolExecutionContext(
        request_id="web-test",
        actor_id="test",
        budget=RequestBudget(max_model_calls=4, max_tool_calls=4, max_estimated_tokens=5000),
        allowed_tools={"web_search"},
    )
    result = registry.execute("web_search", {"query": "latest education policy"}, ctx)

    assert result["results"][0]["title"] == "Example"
    assert result["results"][0]["url"] == "https://example.com"
    assert result["trusted_instruction"].startswith("NONE")
    assert ctx.budget.tool_calls == 1


def test_react_attempts_web_for_explicit_web_request():
    assert ReActToolAgent.should_attempt("Please search the web for the latest education policy.")


def test_multi_agent_passes_adaptive_policy_to_all_review_roles():
    class ScriptedClient:
        default_model = "fake-model"

        def __init__(self):
            self.calls = []

        def chat(self, messages, temperature=0.7, max_completion_tokens=None, **kwargs):
            self.calls.append(messages)
            n = len(self.calls)
            if n == 1:
                return "The learner answered the pending variable check correctly."
            if n == 2:
                return json.dumps({"status": "pass", "feedback": "The learner response was evaluated correctly."})
            return "Correct! y is the variable. The number 4 is the coefficient. Let's continue."

    client = ScriptedClient()
    adaptive = (
        "The learner's latest message is being treated as an answer to your most "
        "recent check question. If correct, explicitly say that the learner is correct."
    )
    result = AgentOrchestrator(
        client,
        manager(),
        max_iterations=1,
    ).run(
        "y",
        [
            {"role": "system", "content": "Context"},
            {"role": "user", "content": "In 4y - 7, which symbol is the variable?"},
            {"role": "assistant", "content": "y"},
        ],
        adaptive_instruction=adaptive,
    )

    assert result["answer"].startswith("Correct!")
    assert all(
        adaptive in "\n".join(m["content"] for m in call)
        for call in client.calls
    )


def test_normal_chat_can_use_a_web_tool_before_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(
        builtins,
        "config",
        SimpleNamespace(
            tavily_api_key="test-key",
            web_search_timeout_seconds=10,
            web_search_max_results=5,
        ),
    )

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"title": "Current source", "url": "https://example.com", "content": "Current result."}
                ]
            }

    monkeypatch.setattr(builtins.requests, "post", lambda *args, **kwargs: FakeResponse())

    class Client:
        default_model = "fake-model"

        def __init__(self):
            self.calls = 0

        def chat(self, messages, temperature=0.7, max_completion_tokens=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({
                    "thought": "The learner explicitly requested a web search.",
                    "action": "web_search",
                    "arguments": {"query": "latest education policy", "max_results": 1},
                    "answer": "",
                })
            assert any("Current source" in str(m["content"]) for m in messages)
            return "I found a current source and can use it to answer your question."

    from week1_llm.security.service import SecurityManager
    from week1_llm.security.audit import AuditLogger

    client = Client()
    pipeline = AssistantPipeline(
        client=client,
        context_manager=manager(),
        security=SecurityManager(audit_logger=AuditLogger(tmp_path / "audit.jsonl")),
    )
    answer = pipeline.chat("Search the web for the latest education policy.")
    assert "current source" in answer
    assert pipeline.last_tool_trace
    assert client.calls == 2


def test_router_common_casual_openers_are_greetings():
    from week1_llm.routing.router import RequestRouter

    router = RequestRouter()
    for message in ("yo", "hey", "sup", "hiya", "howdy", "good morning"):
        decision = router.route(message)
        assert decision.reason == "casual_greeting"


def test_router_preserves_pending_check_for_tutoring_control_request():
    from week1_llm.routing.router import RequestRouter

    decision = RequestRouter().route(
        "Teach me algebra from the beginning",
        adaptive_state={"pending_check": True},
    )
    assert decision.reason == "active_adaptive_control"
    assert decision.activity == ()
