import json

from week1_llm.agents.judge import JudgeAgent
from week1_llm.agents.orchestrator import AgentOrchestrator
from week1_llm.context.window import ContextWindowManager, RegexTokenCounter
from week1_llm.conversation.manager import ConversationManager
from week1_llm.pipeline import AssistantPipeline


def manager():
    return ContextWindowManager(
        RegexTokenCounter(), max_context_tokens=1800, reserved_output_tokens=200
    )


class ScriptedClient:
    default_model = "fake-model"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def chat(self, messages, temperature=0.7, max_completion_tokens=None):
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        })
        if not self.outputs:
            raise AssertionError("Unexpected extra model call")
        return self.outputs.pop(0)


def base_messages(request):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": "Retrieved reference material follows.\nSource: demo"},
        {"role": "user", "content": request},
    ]


def test_judge_parser_fails_closed_on_malformed_output():
    result = JudgeAgent.parse("maybe it is fine")
    assert result.status == "revise"
    assert "not valid structured JSON" in result.feedback


def test_multi_agent_approves_and_returns_trace():
    client = ScriptedClient([
        "Attention uses query, key, and value vectors.",
        json.dumps({"status": "pass", "feedback": "Relevant and sufficiently supported."}),
        "Attention compares queries with keys to weight values.",
    ])
    result = AgentOrchestrator(client, manager(), max_iterations=3).run(
        "Explain attention briefly.", base_messages("Explain attention briefly.")
    )
    assert result["answer"].startswith("Attention compares")
    assert result["review_status"] == "approved"
    assert result["iterations"] == 1
    assert [event["agent"] for event in result["trace"]] == [
        "Orchestrator", "Researcher", "Judge", "Writer", "Orchestrator"
    ]
    assert len(client.calls) == 3


def test_judge_feedback_is_passed_to_researcher_on_retry():
    client = ScriptedClient([
        "Initial notes.",
        json.dumps({"status": "revise", "feedback": "Define the value vector."}),
        "Revised notes: the value vector carries the content.",
        json.dumps({"status": "pass", "feedback": "The missing definition is now present."}),
        "Final explanation includes the value vector.",
    ])
    result = AgentOrchestrator(client, manager(), max_iterations=3).run(
        "Explain Q, K, and V.", base_messages("Explain Q, K, and V.")
    )
    assert result["review_status"] == "approved"
    assert result["iterations"] == 2
    second_researcher_call = client.calls[2]["messages"]
    assert "Define the value vector" in second_researcher_call[-1]["content"]


def test_review_limit_is_reported_without_claiming_approval():
    client = ScriptedClient([
        "Draft one.",
        json.dumps({"status": "revise", "feedback": "Add a caveat."}),
        "Draft two.",
        json.dumps({"status": "revise", "feedback": "Still lacks evidence."}),
        "Cautious final answer.",
    ])
    result = AgentOrchestrator(client, manager(), max_iterations=2).run(
        "Answer carefully.", base_messages("Answer carefully.")
    )
    assert result["review_status"] == "review_limit_reached"
    assert result["iterations"] == 2
    assert any(
        "without Judge approval" in event["summary"]
        for event in result["trace"] if event["agent"] == "Orchestrator"
    )
    assert len(client.calls) == 4


def test_pipeline_multi_agent_stores_only_final_turn():
    client = ScriptedClient([
        "Findings.",
        json.dumps({"status": "pass", "feedback": "Sufficient."}),
        "Final answer.",
    ])
    conversation = ConversationManager(max_turns=5)
    pipeline = AssistantPipeline(
        client=client,
        conversation=conversation,
        context_manager=manager(),
    )
    result = pipeline.multi_agent_chat("Test request.")
    assert result["answer"] == "Final answer."
    assert conversation.as_messages() == [
        {"role": "user", "content": "Test request."},
        {"role": "assistant", "content": "Final answer."},
    ]
