# EduTwin Africa — Curriculum-Aware AI Tutor

EduTwin Africa is a curriculum-aware AI education tutor designed to help learners study through guided explanations, adaptive tutoring, curriculum-grounded retrieval, bounded tool use, and secure multi-agent reasoning.

The project combines the educational LLM foundations developed across Weeks 1–4 with a practical learner-facing application.

It includes:

- Local LLM foundations for educational experimentation
- Groq-based external inference
- South African curriculum knowledge and curriculum-scoped RAG
- State-aware Auto routing
- Adaptive tutoring and learner progress tracking
- Bounded ReAct tool use
- Researcher → Judge → Writer multi-agent collaboration
- Calculator, time, curriculum retrieval, and bounded web-search tools
- Prompt-injection and untrusted-data protections
- Input/output guardrails
- Secret detection and PII auditing
- Rate limiting and request/model/tool budgets
- RAG poisoning defenses
- Session-scoped learner memory
- Streamlit learner interface
- FastAPI backend
- Extensive automated tests and red-team regression coverage

---

## Table of Contents

- [What EduTwin Africa Does](#what-edutwin-africa-does)
- [Architecture](#architecture)
- [Auto Routing](#auto-routing)
- [Curriculum RAG](#curriculum-rag)
- [Adaptive Tutor](#adaptive-tutor)
- [ReAct Tool Use](#react-tool-use)
- [Multi-Agent Collaboration](#multi-agent-collaboration)
- [Security](#security)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Local Setup](#local-setup)
- [Environment Variables](#environment-variables)
- [Run the Application](#run-the-application)
- [Run the Tests](#run-the-tests)
- [FastAPI Backend](#fastapi-backend)
- [Streamlit Community Cloud Deployment](#streamlit-community-cloud-deployment)
- [African Education Context](#african-education-context)
- [Development Workflow](#development-workflow)
- [Future Development](#future-development)
- [Security and Privacy](#security-and-privacy)

---

## What EduTwin Africa Does

EduTwin Africa is designed around a simple learner flow:

```text
Grade
  ↓
Subject
  ↓
Topic
  ↓
Question
  ↓
Adaptive tutoring
  ↓
Explanation / example
  ↓
Understanding check
  ↓
Hint / retry / prerequisite reteach
  ↓
Mastery
  ↓
Next concept
```

The learner-facing application uses **Auto** routing rather than asking learners to select technical agent modes.

The system determines an appropriate path based on the learner's request, current tutoring state, curriculum context, and task complexity.

---

# Architecture

At a high level:

```text
                         EduTwin Africa
                              │
                              ▼
                    Streamlit Learner UI
                              │
                              ▼
                    Assistant Pipeline
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          State-aware Router          Security Layer
                 │                         │
        ┌────────┴────────┐                │
        │                 │                │
        ▼                 ▼                ▼
   Single Agent      Multi-Agent       Guardrails
     + ReAct            + ReAct         Budgets
        │                 │             Rate Limits
        │                 │             Auditing
        │                 │
        │                 ▼
        │          Researcher
        │               ↓
        │             Judge
        │               ↓
        │             Writer
        │
        └───────────────┐
                        ▼
                Allowed Tools
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
    Calculator      Curriculum RAG    Web Search
        │               │                │
        │               ▼                ▼
        │         Local Curriculum   Tavily
        │
        ▼
      Answer
```

The system is deliberately bounded. Tool calls, model calls, request size, estimated tokens, and ReAct steps are subject to configurable limits.

---

# Auto Routing

The main EduTwin interface uses **state-aware Auto routing**.

Learners do not need to choose between Normal Chat and Multi-Agent modes.

```text
Learner request
      |
      v
State-aware Router
      |
      +--> Single Agent + ReAct
      |       |
      |       +--> Calculator
      |       +--> Current Time
      |       +--> Curriculum RAG
      |       +--> Bounded Web Search
      |
      +--> Multi-Agent + ReAct
              |
              +--> Researcher
                      ↓
                    Judge
                      ↓
                    Writer
```

Routing is deterministic and does not require an additional LLM call.

Active adaptive tutoring has priority. This means that a learner response to a pending understanding check, such as `4` or `y`, remains inside the tutoring flow rather than being incorrectly routed as a new research request.

Clearly current, recent, or externally verifiable questions can use bounded web search.

More complex research, comparison, verification, conflicting evidence, or multi-step investigation can escalate to the multi-agent workflow.

Both paths share the same:

- Request-scoped runtime
- Tool registry
- Curriculum source filtering
- ReAct controller
- Security checks
- Request budgets
- Untrusted-data boundaries

The lower-level `chat()` and `multi_agent_chat()` methods remain available for controlled demonstrations and automated tests.

See:

- `AUTO_ROUTING_GUIDE.md`
- `WEB_TOOLS_GUIDE.md`

---

# Curriculum RAG

EduTwin Africa includes a local curriculum knowledge base under:

```text
data/knowledge/
```

The repository contains the curriculum material used by the tutor across Grades 8–12 and supported subjects.

The RAG implementation uses a transparent TF-IDF lexical retrieval baseline.

It is intentionally educational rather than pretending to be a production-scale semantic embedding system.

Key properties include:

- Curriculum-scoped retrieval
- Source filtering
- Versioned in-memory indexes
- Cache invalidation when documents change
- Directory indexing
- Explicit untrusted-data boundaries
- RAG poisoning detection
- Preservation of source text for auditability

The application indexes the curriculum directory at startup.

Programmatic indexing is also supported:

```python
pipeline.rag_index.index_directory(
    path,
    recursive=True,
)
```

The index is rebuilt from the curriculum files when the application starts.

### RAG poisoning protection

Retrieved curriculum content is treated as **untrusted source data**.

Instruction-like content inside retrieved documents is not automatically treated as a trusted system or developer instruction.

This protects the tutor against curriculum-source prompt injection and poisoning attempts while preserving the original source material for inspection and auditing.

See:

```text
src/week1_llm/rag/
tests/test_rag_index.py
```

---

# Adaptive Tutor

EduTwin Africa includes a lightweight, session-scoped adaptive tutoring system.

The core tutoring loop is:

```text
Quick diagnostic
      ↓
Small explanation + example
      ↓
Check understanding
      ↓
Hint / retry / prerequisite reteach
      ↓
Mastery threshold
      ↓
Next concept
```

The adaptive state can track:

- Current concept
- Concepts understood
- Concepts needing practice
- Difficulty
- Recent mistakes
- Pending checks
- Mastery streak

The state is reset when the learner changes the relevant:

```text
Grade + Subject + Topic
```

or starts a new session.

### Adaptive state security

Model-authored state is treated as untrusted data.

The implementation applies validation and bounded handling to model-authored fields and prevents learner-controlled or model-generated state from silently becoming trusted tutoring policy.

Important protections include:

- Bounded text fields
- Control-character handling
- Validated result values
- Semantic boolean parsing
- Application-controlled context identifiers
- Explicit untrusted-state boundaries
- Adversarial regression tests

See:

```text
src/week1_llm/tutoring/adaptive.py
tests/test_adaptive_tutoring.py
WEEK4_ADAPTIVE_TUTOR_GUIDE.md
```

---

# ReAct Tool Use

EduTwin Africa includes a bounded:

```text
Thought → Action → Observation → Answer
```

tool-use loop.

The ReAct controller operates within explicit limits and is not an additional application agent.

Available tools are allowlisted and validated through a tool registry.

The system includes tools such as:

- Calculator
- Current time
- Curriculum retrieval
- Bounded web search

Tool contracts define:

- Tool name
- Description
- Parameters
- JSON Schema
- Safety boundaries

The ReAct controller is bounded by:

```text
REACT_MAX_STEPS
```

The default configuration limits the number of ReAct steps so tool use cannot expand without bound.

See:

```text
TOOLS_GUIDE.md
src/week1_llm/tools/
tests/test_react_tools.py
```

---

# Multi-Agent Collaboration

Complex requests can use the bounded four-agent workflow:

```text
Orchestrator
     │
     ▼
Researcher
     │
     ▼
Judge
     │
     ▼
Writer
     │
     ▼
Final answer
```

### Researcher

Collects and organizes relevant evidence using approved tools and sources.

### Judge

Reviews the research for quality, relevance, grounding, and safety.

The Writer does not simply receive arbitrary research output as trusted truth.

### Writer

Produces the final learner-facing explanation from the approved workflow.

### Orchestrator

Coordinates the bounded workflow and enforces the relevant runtime constraints.

The architecture deliberately avoids uncontrolled autonomous agent loops.

See:

```text
src/week1_llm/agents/
tests/test_multi_agent.py
tests/test_multi_agent_v10.py
MULTI_AGENT_GUIDE.md
```

---

# Security

Security is implemented as a layered system rather than a single prompt.

The security layer includes:

- Input guardrails
- Output guardrails
- Prompt-injection detection
- Secret detection
- PII auditing
- Rate limiting
- Request budgets
- Model-call budgets
- Tool-call budgets
- Estimated-token budgets
- Tool allowlisting
- Tool approval gates
- Bounded ReAct
- Untrusted RAG boundaries
- Untrusted adaptive-state boundaries
- Red-team regression tests
- Audit logging

The application intentionally does not expose:

- Arbitrary code execution
- Destructive tools
- Payments
- Email sending
- Unrestricted browsing

Security-related configuration is controlled through environment variables.

See:

```text
src/week1_llm/security/
src/week1_llm/red_team.py
tests/test_week3_security.py
tests/test_week3_pipeline_security.py
WEEK3_SECURITY_GUIDE.md
```

Run the security regression suite with:

```powershell
python -m week1_llm.red_team
```

---

# Project Structure

```text
EduTwin-Africa/
│
├── digital_twin_streamlit.py       # Primary learner-facing Streamlit UI
├── run.py                          # Local Streamlit launcher
├── web_app.py                      # Local FastAPI launcher
├── main.py
├── app.py
├── calculator_app.py
├── chat_app.py
├── dashboard_app.py
├── todo_app.py
├── smoke_demo.py
│
├── src/
│   └── week1_llm/
│       ├── agents/
│       │   ├── judge.py
│       │   ├── orchestrator.py
│       │   ├── researcher.py
│       │   └── writer.py
│       │
│       ├── api/
│       │   └── groq_client.py
│       │
│       ├── context/
│       │   └── window.py
│       │
│       ├── conversation/
│       │   └── manager.py
│       │
│       ├── curriculum/
│       │   └── catalog.py
│       │
│       ├── digital_twin/
│       │   ├── data.py
│       │   ├── predictor.py
│       │   ├── profile.py
│       │   └── recommendations.py
│       │
│       ├── embeddings/
│       │   ├── embeddings.py
│       │   └── visualization.py
│       │
│       ├── evaluation/
│       │   └── framework.py
│       │
│       ├── foundations/
│       │   ├── neural_network.py
│       │   └── probability.py
│       │
│       ├── language_model/
│       │   ├── inference.py
│       │   └── trainer.py
│       │
│       ├── memory/
│       │   └── memory.py
│       │
│       ├── prompting/
│       │   └── prompts.py
│       │
│       ├── rag/
│       │   └── index.py
│       │
│       ├── routing/
│       │   └── router.py
│       │
│       ├── runtime/
│       │   └── agent_runtime.py
│       │
│       ├── security/
│       │   ├── audit.py
│       │   ├── budget.py
│       │   ├── detectors.py
│       │   ├── exceptions.py
│       │   ├── guardrails.py
│       │   ├── rate_limit.py
│       │   └── service.py
│       │
│       ├── text_processing/
│       │   ├── chunking.py
│       │   ├── loader.py
│       │   └── processor.py
│       │
│       ├── tokenizer/
│       │   └── tokenizer.py
│       │
│       ├── tools/
│       │   ├── builtins.py
│       │   ├── react.py
│       │   ├── registry.py
│       │   └── router.py
│       │
│       ├── transformer/
│       │   ├── attention.py
│       │   ├── block.py
│       │   └── model.py
│       │
│       ├── tutoring/
│       │   └── adaptive.py
│       │
│       └── pipeline.py
│
├── data/
│   ├── knowledge/
│   │   ├── .gitkeep
│   │   └── [curriculum knowledge files]
│   └── README.md
│
├── scripts/
│   ├── run_week1.py
│   ├── run_week2.py
│   └── train_local.py
│
├── tests/
│   ├── test_adaptive_tutoring.py
│   ├── test_multi_agent.py
│   ├── test_rag_index.py
│   ├── test_react_tools.py
│   ├── test_week3_security.py
│   ├── test_ui_theme_and_activity.py
│   └── ...
│
├── web/
│   ├── app.py
│   └── static/
│       └── index.html
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# Requirements

The project requires:

- Python 3.10 or newer
- Git
- A Groq API key for external model inference
- A Tavily API key for bounded web search when web-search functionality is used

The main application uses:

- Python
- Streamlit
- FastAPI
- PyTorch
- NumPy
- scikit-learn
- tiktoken
- pypdf
- requests
- python-dotenv

---

# Local Setup

## 1. Open the project

Open the repository root in VS Code:

```text
EduTwin-Africa/
```

Do not open only the `src/` directory.

---

## 2. Create the virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 3. Install dependencies

With the virtual environment active:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository uses an editable package installation through:

```text
-e .
```

in `requirements.txt`, so a separate `pip install -e .` is normally unnecessary.

---

# Environment Variables

Create a local `.env` file from `.env.example`.

Example:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
WEB_SEARCH_TIMEOUT_SECONDS=10
WEB_SEARCH_MAX_RESULTS=5
```

Additional runtime limits can be configured through the environment, including:

```text
LLM_CONTEXT_TOKEN_BUDGET
LLM_RESPONSE_TOKEN_RESERVE
AGENT_MAX_ITERATIONS
REACT_MAX_STEPS
SECURITY_MAX_INPUT_CHARS
SECURITY_MAX_OUTPUT_CHARS
SECURITY_RATE_LIMIT_REQUESTS
SECURITY_RATE_LIMIT_WINDOW_SECONDS
SECURITY_MAX_MODEL_CALLS
SECURITY_MAX_TOOL_CALLS
SECURITY_MAX_ESTIMATED_TOKENS
```

Never commit `.env`.

The repository contains `.env.example` as a safe template.

---

# Run the Application

## Primary learner application

The primary learner-facing application is:

```text
digital_twin_streamlit.py
```

Run it directly with:

```powershell
streamlit run digital_twin_streamlit.py
```

Or use the convenience launcher:

```powershell
python run.py
```

`run.py` starts the local Streamlit application and opens the browser automatically.

---

## Week 1 foundation demo

The educational Week 1 foundation remains available:

```powershell
python scripts/run_week1.py
```

---

## Local model training smoke test

A deliberately small educational training run is available with:

```powershell
python scripts/train_local.py
```

This is a foundation-learning demonstration and is not intended to train a production-scale LLM.

---

## Week 2 processing/RAG smoke demo

Run:

```powershell
python scripts/run_week2.py
```

---

# Run the Tests

Run the complete automated test suite:

```powershell
python -m pytest -q
```

The test suite covers areas including:

- Attention
- Tokenization
- Probability
- Transformer components
- Context windows
- Curriculum retrieval
- RAG poisoning
- Adaptive tutoring
- Auto routing
- ReAct tools
- Multi-agent workflows
- Groq integration
- Security
- Rate limiting
- Mathematical rendering
- Web mode
- UI behavior
- Week 1/2 integration

The project is designed so that security and regression tests can be run before deployment.

---

# FastAPI Backend

EduTwin Africa also includes a FastAPI backend.

The backend is located at:

```text
web/app.py
```

The local launcher is:

```text
web_app.py
```

Run it with:

```powershell
python web_app.py
```

The local API is available at:

```text
http://127.0.0.1:8000
```

The API exposes normal, Auto, and multi-agent chat paths for controlled backend use and demonstrations.

The primary learner-facing deployment remains the Streamlit application.

---

# Streamlit Community Cloud Deployment

The intended public learner deployment uses:

```text
GitHub
   ↓
Streamlit Community Cloud
   ↓
digital_twin_streamlit.py
```

## 1. Push the repository to GitHub

The GitHub repository is:

```text
EduTwin-Africa
```

The curriculum files under:

```text
data/knowledge/
```

must remain part of the repository because the RAG system loads them at application startup.

---

## 2. Create the Streamlit application

In Streamlit Community Cloud:

1. Connect the GitHub account.
2. Select the `EduTwin-Africa` repository.
3. Select the `main` branch.
4. Set the main application file to:

```text
digital_twin_streamlit.py
```

5. Deploy the application.

---

## 3. Configure secrets

Do **not** upload `.env` to GitHub.

Instead, configure the production secrets in Streamlit Community Cloud.

At minimum:

```toml
GROQ_API_KEY = "your_groq_api_key"
TAVILY_API_KEY = "your_tavily_api_key"
WEB_SEARCH_TIMEOUT_SECONDS = "10"
WEB_SEARCH_MAX_RESULTS = "5"
```

Only add additional configuration values when required.

API keys should never be placed directly in Python source code, README files, screenshots, or Git commits.

---

## 4. Deployment entry point

The primary deployment entry point is:

```text
digital_twin_streamlit.py
```

The following files remain useful for local development and backend demonstrations:

```text
run.py
web_app.py
web/app.py
```

---

# African Education Context

EduTwin Africa is designed with practical African education constraints in mind.

The architecture supports:

- Curriculum-grounded learning
- South African Grades 8–12 content
- Multiple school subjects
- Low-bandwidth-conscious application design
- Bounded external web usage
- Local curriculum retrieval
- Session-based learner progress
- Adaptive explanations
- Clear learner-facing instructions

The application is designed so that curriculum information can remain available locally rather than requiring an external web request for every educational question.

This is particularly useful in environments where connectivity, bandwidth, latency, or data costs can be constraints.

---

# Development Workflow

A recommended development workflow is:

```text
1. Make a focused change
        ↓
2. Run relevant tests
        ↓
3. Run the complete test suite
        ↓
4. Review security implications
        ↓
5. Test the Streamlit application locally
        ↓
6. Commit to Git
        ↓
7. Push to GitHub
        ↓
8. Streamlit Community Cloud redeploys
        ↓
9. Test the deployed application
```

For changes affecting:

- Routing
- RAG
- Adaptive state
- Security
- Tools
- Multi-agent behavior

run the relevant regression tests as well as the full suite.

---

# Future Development

The current architecture is functional, but there are several potential future extensions.

Possible improvements include:

- Improved/subword BPE tokenization
- Larger and cleaner training corpora
- Train/validation/test splits
- More efficient batching and data loading
- Stronger embedding training
- More advanced multi-head attention
- Positional encoding/embeddings
- Additional transformer blocks
- Improved normalization and dropout strategies
- Optimizers and learning-rate schedules
- Checkpointing and evaluation improvements
- More advanced autoregressive inference
- Improved sampling
- Model/version configuration
- Expanded multilingual support
- Additional educational tools
- Further curriculum coverage
- More advanced semantic retrieval
- Additional accessibility improvements
- More extensive low-bandwidth/mobile optimization

These are future extensions rather than prerequisites for the current application architecture.

---

# Security and Privacy

EduTwin Africa is designed with explicit security boundaries.

Never commit:

```text
.env
API keys
access tokens
private credentials
local secrets
```

The repository `.gitignore` excludes local secrets and development artifacts.

If an API key has ever been placed directly inside a source file, committed to Git, or shared publicly, rotate that key immediately.

The application also applies runtime security controls including:

- Input validation
- Output validation
- Prompt-injection detection
- Secret detection
- PII auditing
- Rate limiting
- Request budgets
- Model-call limits
- Tool-call limits
- Estimated-token limits
- Tool allowlisting
- Bounded ReAct
- RAG poisoning protection
- Adaptive-state trust boundaries
- Audit logging

Security is treated as part of the application architecture rather than as a final deployment step.

---

# Educational Foundation

EduTwin Africa grew from an educational LLM foundation covering concepts such as:

```text
Probability
    ↓
Conditional Probability / Markov Prediction
    ↓
Neural Networks
    ↓
Digital Twin Prediction
    ↓
Tokenization + Vocabulary
    ↓
Word Embeddings
    ↓
Similarity / PCA / Clustering
    ↓
Language Model Concepts
    ↓
Q / K / V Attention
    ↓
Causal Transformer Block
    ↓
Small Local Language Model
    ↓
Prompt Construction
    ↓
Groq API Integration
    ↓
Conversation + Memory
    ↓
Text Processing + Context Management
    ↓
Curriculum RAG
    ↓
Tool Use + ReAct
    ↓
Security + Red Teaming
    ↓
Adaptive Tutoring
    ↓
EduTwin Africa Application
```

The local model components remain intentionally small and educational.

The Groq integration is an external inference provider and is kept conceptually separate from the local model foundations.

This distinction allows the project to demonstrate both:

1. How LLM foundations can be implemented and studied locally.
2. How those foundations can be combined with modern hosted inference to build a practical education application.

---

# Documentation

Additional architecture and development documentation is available in:

```text
ARCHITECTURE.md
AUTO_ROUTING_GUIDE.md
WEB_TOOLS_GUIDE.md
MULTI_AGENT_GUIDE.md
TOOLS_GUIDE.md
WEEK3_SECURITY_GUIDE.md
WEEK4_EDUTWIN_GUIDE.md
WEEK4_ADAPTIVE_TUTOR_GUIDE.md
```

These documents provide deeper explanations of individual subsystems and course-development stages.

---

# License

Add the project's chosen license here before publishing if a specific open-source license is required.

---

## EduTwin Africa

A curriculum-aware AI tutor combining LLM foundations, curriculum RAG, adaptive learning, bounded agents, tool use, and security into a practical educational application.