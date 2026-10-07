# EduTwin Africa — Week 4 Implementation Guide

## Product direction

EduTwin Africa is a curriculum-aware educational tutor. The learner flow is:

**Grade → Subject → Topic → Question → EduTwin teaches → Follow-up**

Grades 8–12 are available in the UI. Subjects and topics are discovered from the `.txt`
curriculum files in `data/knowledge`, so the application does not invent curriculum
coverage that is not present in the project.

## Week 4 frontend

The Streamlit application is `digital_twin_streamlit.py`.

It provides:

- **Chat** — the main tutoring experience, real LLM calls, curriculum-scoped RAG,
  loading/error states, feedback, and optional agent/tool trace.
- **History** — completed learning sessions captured when a learner starts a new session.
- **Progress** — real message, response-time, tool-use, feedback, session, and security
  metrics.
- **Settings** — response language, creativity, workflow mode, appearance, export, and
  privacy/safety information.
- **Sidebar** — learner profile and Grade → Subject → Topic learning context.

## Session isolation

Streamlit learner state is session-scoped:

- conversation history is held by that learner's `ConversationManager`
- learner memory is in-memory and is not written to `data/memory.json`
- a per-session actor ID is supplied to security/rate-limit checks
- the RAG index is shared as static reference data only

This avoids the earlier risk of a globally cached mutable conversation/memory object
being shared between Streamlit users.

## Curriculum-aware RAG

The curriculum catalog reads metadata from the knowledge-base `.txt` files. The selected
Grade + Subject + Topic is converted into an exact source filter for RAG retrieval.

The same source filter is applied to the Week 3 `knowledge_search` tool during ReAct
execution.

Retrieved curriculum content remains untrusted reference data. It is never promoted to
system/developer instruction.

## Existing Week 1–3 engine preserved

The project retains:

- Groq API client
- context-window management
- conversation manager
- RAG
- memory
- Researcher → Judge → Writer
- bounded ReAct Thought → Action → Observation → Answer
- tool registry with name/description/JSON-schema parameters
- input/output guardrails
- prompt-injection and secret detection
- rate limiting
- model/tool/token budgets
- audit logging
- red-team tests

The new Week 4 layer adds application/UI behavior without removing the Week 3 security
boundary.

## Run commands

Main EduTwin application:

```powershell
python run.py
```

This starts Streamlit and opens the local browser.

Foundation smoke demo:

```powershell
python smoke_demo.py
```

FastAPI backend/API:

```powershell
python web_app.py
```

Tests:

```powershell
python -m pytest -q
```

## Knowledge-base expansion

Add curriculum files under:

```text
data/knowledge/
```

Recommended metadata at the top of each file:

```text
GRADE 9 MATHEMATICS
SUBJECT: MATHEMATICS
TOPIC: FUNCTIONS
```

The UI will discover the new Grade/Subject/Topic coverage automatically after the
application restarts.


## Adaptive tutoring

The Week 4 tutor now follows a small-step learning loop:

**quick diagnostic → explain → example → check → adapt**

When a learner struggles, the intended sequence is:

**hint → retry → prerequisite reteach → retry**

EduTwin keeps only a lightweight session-scoped learning state: current concept,
concepts understood, concepts needing practice, difficulty, recent mistakes, pending
checks, and a mastery streak. It normally looks for 2–3 successful checks of increasing
difficulty before moving on, while allowing a clear explanation to demonstrate mastery.

See `WEEK4_ADAPTIVE_TUTOR_GUIDE.md` and
`src/week1_llm/tutoring/adaptive.py`.


## Auto agent routing

The learner-facing experience uses a state-aware Auto router. Active adaptive checks take priority; ordinary tutoring uses Single-Agent + ReAct, current or externally verifiable requests can use bounded web search, and complex research/comparison/verification can escalate to the Researcher → Judge → Writer workflow. Both paths share the same request-scoped tool runtime and security controls.
