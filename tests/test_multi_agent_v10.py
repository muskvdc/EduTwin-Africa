from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")

def test_judge_allows_incremental_adaptive_lessons():
    text = read("src/week1_llm/agents/judge.py")
    assert "adaptive tutor" in " ".join(text.lower().split())
    normalized = " ".join(text.lower().split())
    assert "focused," in normalized and "accurate first" in normalized and "lesson" in normalized
    assert "does not need" in normalized and "entire subject" in normalized

def test_researcher_supports_incremental_first_lesson():
    text = read("src/week1_llm/agents/researcher.py")
    assert "incremental first lesson" in " ".join(text.lower().split())
    assert "teach me from the beginning" in " ".join(text.lower().split())

def test_multi_agent_default_review_limit_is_bounded():
    text = read("src/week1_llm/config.py")
    assert 'agent_max_iterations: int = _int_env("AGENT_MAX_ITERATIONS", 1)' in text

def test_writer_requires_judge_approval():
    text = read("src/week1_llm/agents/orchestrator.py")
    assert 'if approved:' in text
    assert 'self.writer.run(' in text
    assert 'no unapproved answer was shown to the learner' in text
