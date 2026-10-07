# EduTwin Africa — Adaptive Tutor Upgrade

## What changed

EduTwin now uses a lightweight, session-scoped adaptive learning state.

The tutoring loop is:

**quick diagnostic → small lesson → check → adapt → retry or progress**

### Behaviour decisions

- **Hybrid start:** a learner receives a short introduction followed by one quick diagnostic/check question.
- **When the learner struggles:** EduTwin uses **hint → retry → prerequisite reteach** rather than immediately giving the answer.
- **Learning state:** only lightweight session state is tracked:
  - current concept
  - concepts understood
  - concepts needing practice
  - current difficulty
  - recent mistakes
  - whether a check is waiting for an answer
  - mastery streak
- **Mastery:** normally requires 2–3 successful checks of increasing difficulty, while allowing the tutor to recognise a clear demonstration of understanding.
- The state resets when the learner changes Grade + Subject + Topic or starts a new session.

## Architecture

The adaptive state lives in:

```text
src/week1_llm/tutoring/adaptive.py
```

The Streamlit application keeps the state in the learner's Streamlit session. It is not written to the persistent memory file.

The LLM returns a small application metadata block after each response. The pipeline parses and removes that block before the learner sees the response or before the response is stored in conversation history.

The metadata is treated as untrusted model output and is validated/clamped before it becomes application state.

## Why this is useful for the course

This upgrade demonstrates an important LLM application pattern:

```text
LLM response
     ↓
evaluate learner evidence
     ↓
update lightweight learning state
     ↓
include state in next prompt
     ↓
adapt the next teaching step
```

It does **not** create a second AI model, a permanent student profile, or a separate course-generating agent.

## Testing

The project includes unit tests for:

- learning-state reset and round-trip handling
- adaptive tutoring instructions
- metadata extraction and removal

The full project test suite was run after this upgrade.


## Auto agent routing

The learner-facing experience uses a state-aware Auto router. Active adaptive checks take priority; ordinary tutoring uses Single-Agent + ReAct, current or externally verifiable requests can use bounded web search, and complex research/comparison/verification can escalate to the Researcher → Judge → Writer workflow. Both paths share the same request-scoped tool runtime and security controls.
