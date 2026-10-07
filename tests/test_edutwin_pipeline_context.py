from week1_llm.context.window import ContextWindowManager, RegexTokenCounter
from week1_llm.memory.memory import Memory
from week1_llm.pipeline import AssistantPipeline
from week1_llm.rag.index import RAGIndex


class FakeClient:
    default_model = "fake-model"

    def chat(self, messages, temperature=0.7, max_completion_tokens=None):
        return "safe answer"


def manager():
    return ContextWindowManager(
        RegexTokenCounter(), max_context_tokens=1800, reserved_output_tokens=200
    )


def test_pipeline_can_scope_rag_to_selected_curriculum_source():
    rag = RAGIndex()
    rag.add_document("algebra", "grade8_mathematics_algebra.txt", "brackets and equations")
    rag.add_document("geometry", "grade8_mathematics_geometry.txt", "angles and shapes")

    pipeline = AssistantPipeline(
        client=FakeClient(),
        memory=Memory(persist=False),
        rag_index=rag,
        context_manager=manager(),
    )

    messages = pipeline.build_messages(
        "How do I solve equations with brackets?",
        source_filter=["grade8_mathematics_algebra.txt"],
    )

    reference_messages = [
        message["content"]
        for message in messages
        if message["role"] == "user"
        and "retrieved_reference_untrusted_data" in message["content"]
    ]
    assert reference_messages
    assert "grade8_mathematics_algebra.txt" in reference_messages[0]
    assert "grade8_mathematics_geometry.txt" not in reference_messages[0]


def test_memory_can_be_session_only():
    memory = Memory(persist=False)
    memory.set("learner_name", "Test Learner")

    assert memory.get("learner_name") == "Test Learner"
    assert not memory.path.exists()
