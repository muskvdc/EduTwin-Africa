# EduTwin Africa — Auto Routing Architecture

## Learner experience

The learner-facing application uses **Auto** routing only. Learners do not
choose between Normal Chat and Multi-Agent.

For each request, a deterministic state-aware router selects:

- **Single-Agent + ReAct** for ordinary tutoring and tool-assisted requests.
- **Single-Agent + ReAct + web search** for clearly current, recent, or
  externally verifiable information.
- **Multi-Agent + ReAct** for complex research, comparison, verification,
  conflicting evidence, or multi-step investigation.

Active adaptive tutoring has priority. When a learner is answering a pending
check question, the router keeps the request in the tutoring path so short
answers such as `4` or `y` are evaluated as learner answers rather than being
misclassified as generic tool requests.

## Shared runtime

Both execution paths use the same request-scoped `AgentRuntime`:

1. build the allowlisted tool registry;
2. apply the selected curriculum source filter;
3. create the bounded tool execution context;
4. run the same ReAct controller when a tool is warranted;
5. pass tool observations into the final model as untrusted reference data.

This avoids maintaining separate tool implementations for Normal Chat and
Multi-Agent.

## Multi-Agent escalation

Complex requests use the existing:

`Researcher → Judge → Writer`

workflow after the shared ReAct stage. The Judge still has to approve the
research findings before the Writer produces learner-facing content.

## Learner-visible activity

The Streamlit UI shows a subtle activity indicator rather than exposing the
full internal pipeline:

`Researching → Checking → Preparing your explanation`

For ordinary requests it may simply show:

`Preparing your explanation`

The detailed trace remains available in the existing activity expander for
debugging/course demonstration.

## Course/developer demonstration

The backend `chat()` and `multi_agent_chat()` methods remain available for
tests and controlled course demonstrations. They are not exposed as a learner
choice in the main EduTwin interface.

## Routing is intentionally conservative

The router does not use an extra LLM call. This keeps routing cheap,
predictable, and easy to evaluate. It is deliberately biased toward the
single-agent path unless the request clearly benefits from multi-agent
research/review.

Web search is not triggered merely because a question could benefit from
the internet. Clearly current/recent/time-sensitive requests can use the web
tool, while ordinary curriculum questions remain grounded in the local
curriculum pack first.
