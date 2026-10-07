# Secure Tool Layer + ReAct Tool Use

Week 3 now has two related layers:

1. **Tool contract** — every tool exposes a clear name, diligent description, and
   JSON Schema parameters.
2. **ReAct controller** — the model can use those tools through a bounded
   Thought → Action → Observation → Answer loop.

## Tool contract

Each registered tool has:

- **Name** — the exact identifier the agent may request.
- **Description** — what the tool does, when it is useful, and important safety
  boundaries.
- **Parameters** — a JSON Schema object describing accepted arguments, types,
  required fields, and parameter meaning.

For example, the calculator contract is conceptually:

```text
name: calculator

description:
  Evaluate a mathematical expression with a restricted arithmetic parser.
  It supports numeric arithmetic and never executes Python code.

parameters:
  expression
    type: string
    required: yes
    description: the numeric expression to evaluate
```

The registry also keeps the trusted Python handler, input validator, risk level,
approval gate and enabled/disabled state. The model never receives direct
access to those handlers.

## ReAct loop

The Multi-Agent mode now performs bounded tool use before the four-agent review
workflow:

```text
User request
     ↓
Thought
     ↓
Action
     ↓
Tool executes
     ↓
Observation
     ↓
Thought
     ↓
Action / Answer
     ↓
Researcher
     ↓
Judge
     ↓
Writer
     ↓
Final answer
```

The ReAct controller is a **tool-use controller, not a fifth application
agent**. The core four-agent architecture remains:

```text
Orchestrator → Researcher → Judge → Writer
```

The ReAct trace is visible in the browser so the Week 3 lesson can be observed
directly. Thought entries are concise rationale summaries rather than private
chain-of-thought transcripts.

## Security boundary

The `ToolRegistry` remains the enforcement point:

- explicit allowlist
- schema/argument validation
- per-request tool budget
- risk classification
- approval gates
- JSON-serializable tool results
- request IDs for audit correlation

The model may request `calculator`, for example, but it cannot execute arbitrary
Python code.

The default tools are intentionally low-risk:

- `calculator`
- `current_time`
- `knowledge_search`
- `memory_lookup`

`knowledge_search` and `memory_lookup` return **untrusted data**, not
instructions.

A deterministic `SafeToolRouter` remains in the codebase as a simple comparison
implementation for the earlier Week 3 security lesson, but the active
Multi-Agent pipeline now uses the ReAct controller.

## Model-call budget

Because ReAct adds bounded tool-selection calls to the existing Researcher →
Judge review loop, the default multi-agent model-call budget is **10**. With
`REACT_MAX_STEPS=2` and `AGENT_MAX_ITERATIONS=3`, this gives a hard ceiling for
the combined tool-use, review and writing workflow.

Do not increase the budget casually. Resource limits are part of the security
design.
