# Week 3 — Security, Safety, Guardrails and Red Teaming

This upgrade applies the four Week 3 exercise themes to the existing Week 1–2 + multi-agent project:

1. Tool use
2. Evaluation/testing
3. Guardrails/safety
4. Red teaming/security

## Security architecture

```text
User
  |
  v
Input Guardrails
  |-- size limit
  |-- prompt-injection heuristics
  |-- secret detection
  |-- PII auditing
  |-- rate limit
  v
Assistant Pipeline
  |
  +--> RAG / memory are explicitly UNTRUSTED DATA
  |
  +--> Multi-Agent workflow
  |      Researcher -> Judge loop -> Writer
  |
  +--> Allowlisted low-risk tools
  |      calculator
  |      current_time
  |      knowledge_search
  |      memory_lookup
  |
  v
Output Guardrails
  |-- output-size limit
  |-- secret-leak detection
  |-- system-prompt fragment detection
  v
User
```

## Important security principles

### Prompts are not security boundaries

The system prompt tells the model not to follow instructions inside user content, retrieved documents, memory, or tool results. That is defense-in-depth, not a security guarantee.

The stronger control is architectural separation: retrieved documents, memory and tool observations are placed in user/data content rather than being promoted to system-message authority.

### Tool execution is allowlisted

The tool registry validates tool names and arguments, tracks risk level, supports approval gates, and enforces a per-request tool-call budget.

The calculator never uses Python `eval`; it parses a restricted arithmetic AST.

Side-effecting tools should not be exposed until authentication, authorization, approval, audit and idempotency requirements exist.

### PII is audited, not blindly blocked

Users may legitimately ask about their own personal information. The detector therefore records PII categories for security telemetry instead of automatically censoring every message containing PII.

High-confidence credential patterns are blocked.

### Rate limits and budgets

The project now has:
- request rate limiting
- model-call limits
- tool-call limits
- estimated token budgets
- context/output size limits

The budgets are safety controls, not exact billing meters. Provider-reported usage should be added when a production deployment needs exact cost accounting.

### Red teaming

Run:

```powershell
python -m week1_llm.red_team
```

The suite tests direct instruction override, system-prompt extraction, fake system messages, delimiter spoofing, safety bypass language, tool-policy bypass and oversized inputs.

A security score is not proof of safety. New attacks should be added whenever a failure is discovered.

## Known limitations

This remains a local educational project, not a complete security certification:

- Rate limiting is in-memory and single-process.
- There is no real authentication or per-user authorization layer.
- The prompt-injection detector is heuristic.
- No model-based moderation classifier is installed.
- Exact provider billing is not measured because the current HTTP client does not expose provider usage in its return value.
- The safe tool router uses deterministic rules rather than provider-native function calling.
- External web browsing, email sending, payments, code execution and destructive database tools remain intentionally unavailable.
- If such capabilities are added later, they require explicit identity, authorization, approval, timeout, idempotency and audit controls.

The goal is to establish sound boundaries without pretending that a collection of regexes makes an agent secure.


## Tool Use: ReAct loop

The tool layer now also demonstrates the Week 3 ReAct pattern:

**Thought → Action → Observation → Answer**

The Thought is a short rationale summary for educational visibility, not a
private chain-of-thought transcript. The Action selects an allowlisted tool and
provides schema-validated arguments. The Observation is the tool result,
treated as data rather than instructions. The Answer is the tool-use
conclusion handed to the Researcher before the existing Judge/Writer workflow.

The four application agents remain unchanged:

```text
Orchestrator → Researcher → Judge → Writer
```

ReAct is a bounded tool-use controller inside that architecture, not a fifth
agent.

Every tool exposes an explicit contract containing:

1. name
2. diligent description of what the tool does
3. JSON Schema parameters, including argument types, required fields and
   parameter descriptions

The registry remains the security enforcement point even if provider-native
function calling is added later.
