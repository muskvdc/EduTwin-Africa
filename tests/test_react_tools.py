import json

from week1_llm.context.window import ContextWindowManager, RegexTokenCounter
from week1_llm.memory.memory import Memory
from week1_llm.rag.index import RAGIndex
from week1_llm.security.budget import RequestBudget, BudgetedClient
from week1_llm.tools import ReActToolAgent, ToolExecutionContext, build_default_tool_registry


class SequenceClient:
    default_model = "fake-model"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def chat(self, messages, temperature=0.7, max_completion_tokens=None, **kwargs):
        self.calls.append(messages)
        return self.outputs.pop(0)


def manager():
    return ContextWindowManager(
        RegexTokenCounter(), max_context_tokens=1800, reserved_output_tokens=300
    )


def context():
    return ToolExecutionContext(
        request_id="react-test",
        actor_id="test",
        budget=RequestBudget(max_model_calls=10, max_tool_calls=4, max_estimated_tokens=10000),
        allowed_tools={"calculator", "current_time", "knowledge_search", "memory_lookup"},
    )


def test_tool_contract_has_name_description_and_parameters():
    registry = build_default_tool_registry(RAGIndex(), Memory())
    tools = {item["name"]: item for item in registry.list_tools()}

    assert {"calculator", "current_time", "knowledge_search", "memory_lookup"} <= tools.keys()
    for tool in tools.values():
        assert tool["name"]
        assert len(tool["description"]) > 20
        assert tool["parameters"]["type"] == "object"
        assert "properties" in tool["parameters"]

    calculator = tools["calculator"]
    assert calculator["parameters"]["properties"]["expression"]["type"] == "string"
    assert "expression" in calculator["parameters"]["required"]


def test_registry_exposes_openai_compatible_tool_schema():
    registry = build_default_tool_registry()
    schemas = registry.model_tool_schemas()

    assert all(item["type"] == "function" for item in schemas)
    calculator = next(item for item in schemas if item["function"]["name"] == "calculator")
    assert calculator["function"]["description"]
    assert calculator["function"]["parameters"]["required"] == ["expression"]


def test_react_loop_records_thought_action_observation_answer():
    client = SequenceClient([
        json.dumps({
            "thought": "The request is arithmetic, so the calculator is appropriate.",
            "action": "calculator",
            "arguments": {"expression": "15 * 4"},
            "answer": "",
        }),
        json.dumps({
            "thought": "The calculator returned the needed result.",
            "action": "answer",
            "arguments": {},
            "answer": "The calculation result is 60.",
        }),
    ])
    ctx = context()
    budget = ctx.budget
    guarded = BudgetedClient(client, budget)
    agent = ReActToolAgent(guarded, manager(), build_default_tool_registry(), max_steps=2)

    result = agent.run("What is 15 * 4?", ctx)

    statuses = [event["status"] for event in result.trace]
    assert statuses == ["thought", "action", "observation", "thought", "answer"]
    assert result.observations[0]["result"]["result"] == 60
    assert result.answer == "The calculation result is 60."
    assert budget.model_calls == 2
    assert budget.tool_calls == 1


def test_react_loop_cannot_execute_unregistered_tool():
    client = SequenceClient([
        json.dumps({
            "thought": "I will try an unavailable capability.",
            "action": "run_python",
            "arguments": {"code": "print(1)"},
            "answer": "",
        }),
        json.dumps({
            "thought": "The unavailable action was rejected.",
            "action": "answer",
            "arguments": {},
            "answer": "No allowed tool was needed.",
        }),
    ])
    ctx = context()
    agent = ReActToolAgent(
        BudgetedClient(client, ctx.budget),
        manager(),
        build_default_tool_registry(),
        max_steps=1,
    )

    result = agent.run("Do something unsafe.", ctx)

    assert result.observations == []
    assert any("Tool action 'run_python' was not executed" in e["summary"] for e in result.trace)


def test_knowledge_search_respects_curriculum_source_filter():
    rag = RAGIndex()
    rag.add_document("algebra", "grade8_mathematics_algebra.txt", "equations with brackets")
    rag.add_document("geometry", "grade8_mathematics_geometry.txt", "angles and shapes")
    registry = build_default_tool_registry(
        rag,
        Memory(persist=False),
        knowledge_source_filter=["grade8_mathematics_algebra.txt"],
    )
    ctx = context()
    result = registry.execute(
        "knowledge_search",
        {"query": "equations brackets", "top_k": 4},
        ctx,
    )
    assert result["results"]
    assert all(
        item["source"] == "grade8_mathematics_algebra.txt"
        for item in result["results"]
    )
