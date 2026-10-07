import pytest

from week1_llm.context.window import ContextBudgetError, ContextWindowManager, RegexTokenCounter


def manager(budget=100, reserve=10):
    return ContextWindowManager(
        RegexTokenCounter(),
        max_context_tokens=budget,
        reserved_output_tokens=reserve,
    )


def test_context_keeps_system_latest_user_and_recent_turns():
    window = manager(80, 10)
    messages = [
        {"role": "system", "content": "Always be accurate."},
        {"role": "user", "content": "Old question with many details"},
        {"role": "assistant", "content": "Old answer with many details"},
        {"role": "user", "content": "Recent question"},
        {"role": "assistant", "content": "Recent answer"},
        {"role": "user", "content": "Current question"},
    ]
    result = window.fit_messages(messages)
    assert result[0] == messages[0]
    assert result[-1] == messages[-1]
    assert any(m["content"] == "Recent question" for m in result)
    assert window.count_messages(result) <= window.input_budget


def test_context_raises_instead_of_dropping_protected_system_or_latest_user():
    window = manager(10, 2)
    messages = [
        {"role": "system", "content": "Important instructions that cannot fit"},
        {"role": "user", "content": "Do this"},
    ]
    with pytest.raises(ContextBudgetError):
        window.fit_messages(messages)


def test_context_rejects_unknown_roles():
    with pytest.raises(ValueError, match="Unsupported"):
        manager().fit_messages([
            {"role": "system", "content": "Rules"},
            {"role": "tool", "content": "Tool output"},
            {"role": "user", "content": "Question"},
        ])


def test_tokenizer_adapter_counts_deterministically():
    counter = RegexTokenCounter()
    assert counter.count("hello, world!") == 4


def test_optional_retrieval_context_is_dropped_before_protected_instructions():
    window = manager(35, 5)
    messages = [
        {"role": "system", "content": "Never ignore these core rules."},
        {"role": "system", "content": "Large retrieved source material " * 30},
        {"role": "user", "content": "Answer my question."},
    ]
    result = window.fit_messages(messages, optional_system_indices={1})
    assert result == [messages[0], messages[2]]
    assert window.count_messages(result) <= window.input_budget
