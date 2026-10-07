from types import SimpleNamespace

from week1_llm.context.window import ContextWindowManager, RegexTokenCounter
from week1_llm.pipeline import AssistantPipeline
from week1_llm.rag import RAGIndex


class FakeClient:
    default_model = "fake-model"

    def chat(self, messages, temperature=0.7, max_completion_tokens=None):
        assert max_completion_tokens == 20
        assert any(m["role"] == "system" for m in messages)
        return "A test response."


class FakeMemory:
    def as_system_context(self):
        return "Known user context:\n- prefers concise explanations"


def test_pipeline_adds_rag_context_and_bounds_messages():
    rag = RAGIndex()
    rag.add_document("guide", "kb://guide", "Attention uses query, key, and value vectors.")
    context = ContextWindowManager(
        RegexTokenCounter(),
        max_context_tokens=300,
        reserved_output_tokens=20,
    )
    pipeline = AssistantPipeline(
        client=FakeClient(),
        memory=FakeMemory(),
        context_manager=context,
        rag_index=rag,
    )
    messages = pipeline.build_messages("Explain query key value attention.")
    assert messages[0]["role"] == "system"
    # Week 3 security hardening: memory and RAG are untrusted data, not system instructions.
    assert "Known user context" not in messages[0]["content"]
    assert not any(
        m["role"] == "system" and "kb://guide" in m["content"]
        for m in messages
    )
    assert any(
        m["role"] == "user"
        and "Known user context" in m["content"]
        and "kb://guide" in m["content"]
        for m in messages[:-1]
    )
    assert messages[-1] == {"role": "user", "content": "Explain query key value attention."}
    assert context.count_messages(messages) <= context.input_budget
    assert pipeline.chat("Explain query key value attention.") == "A test response."


def test_tiktoken_adapter_uses_model_encoding_or_fallback(monkeypatch):
    class FakeEncoding:
        def encode(self, text, disallowed_special=()):
            return text.split()

    calls = []
    def encoding_for_model(model):
        raise KeyError(model)
    def get_encoding(name):
        calls.append(name)
        if name == "o200k_harmony":
            return FakeEncoding()
        raise ValueError(name)

    fake_module = SimpleNamespace(
        encoding_for_model=encoding_for_model,
        get_encoding=get_encoding,
    )
    monkeypatch.setitem(__import__("sys").modules, "tiktoken", fake_module)

    from week1_llm.context.window import TiktokenTokenCounter
    counter = TiktokenTokenCounter("openai/gpt-oss-120b")
    assert counter.count("one two three") == 3
    assert calls == ["o200k_harmony"]
