import json

from week1_llm.context.window import ContextWindowManager, RegexTokenCounter
from week1_llm.pipeline import AssistantPipeline
from week1_llm.routing import RequestRouter


def manager():
    return ContextWindowManager(
        RegexTokenCounter(), max_context_tokens=1800, reserved_output_tokens=200
    )


def test_router_prioritizes_pending_adaptive_check():
    decision = RequestRouter().route(
        "4",
        adaptive_state={"pending_check": True, "current_concept": "coefficient"},
    )
    assert decision.route == "single_agent"
    assert decision.reason == "active_adaptive_check"


def test_router_keeps_current_information_single_agent():
    decision = RequestRouter().route("What is the latest South African education policy?")
    assert decision.route == "single_agent"
    assert decision.reason == "time_sensitive_or_web_request"


def test_router_escalates_complex_research():
    decision = RequestRouter().route(
        "Research the latest policy, compare several sources, verify the claims, "
        "and explain the differences."
    )
    assert decision.route == "multi_agent"
    assert decision.reason == "complex_research_or_verification"
    assert list(decision.activity) == [
        "Researching",
        "Checking",
        "Preparing your explanation",
    ]


def test_router_defaults_to_single_agent():
    decision = RequestRouter().route("Explain quadratic equations simply.")
    assert decision.route == "single_agent"
    assert decision.reason == "standard_tutoring_request"


class ScriptedClient:
    default_model = "fake-model"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def chat(self, messages, temperature=0.7, max_completion_tokens=None, **kwargs):
        self.calls.append(messages)
        if not self.outputs:
            raise AssertionError("Unexpected extra model call")
        return self.outputs.pop(0)


def test_auto_chat_uses_single_agent_for_normal_tutoring():
    client = ScriptedClient(["A variable is a symbol that can represent a number."])
    pipeline = AssistantPipeline(client=client, context_manager=manager())

    result = pipeline.auto_chat(
        "Explain variables simply.",
        adaptive_state={"pending_check": False},
    )

    assert result["route"] == "single_agent"
    assert result["route_reason"] == "standard_tutoring_request"
    assert result["answer"].startswith("A variable")
    assert len(client.calls) == 1


def test_auto_chat_escalates_complex_research_to_multi_agent():
    client = ScriptedClient([
        "The research findings compare the requested sources.",
        json.dumps({"status": "pass", "feedback": "Sufficiently supported."}),
        "Here is the checked comparison.",
    ])
    pipeline = AssistantPipeline(client=client, context_manager=manager())

    result = pipeline.auto_chat(
        "Research and compare several sources about this topic.",
        adaptive_state={"pending_check": False},
    )

    assert result["route"] == "multi_agent"
    assert result["review_status"] == "approved"
    assert result["answer"].startswith("Here is the checked")
    assert len(client.calls) == 3


def test_adaptive_policy_is_not_treated_as_protected_prompt_fragment():
    client = ScriptedClient([
        "Correct! The learner's latest answer is correct, so continue with the next small check."
    ])
    pipeline = AssistantPipeline(client=client, context_manager=manager())

    result = pipeline.chat(
        "4",
        extra_system_instruction=(
            "ADAPTIVE TUTORING POLICY: If correct, explicitly say that the "
            "learner is correct and briefly explain why."
        ),
    )

    assert result.startswith("Correct!")
