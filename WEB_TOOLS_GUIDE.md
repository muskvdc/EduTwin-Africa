# EduTwin v11 Tool Use

This patch adds real request-scoped tool use to Normal Chat and Multi-Agent mode.

## Web search

Web search is provided through Tavily and is optional. Add these lines to the project's existing `.env` file:

TAVILY_API_KEY=your_tavily_api_key
WEB_SEARCH_TIMEOUT_SECONDS=10
WEB_SEARCH_MAX_RESULTS=5

Restart EduTwin after changing `.env`.

Do not paste the API key into chat.

## What EduTwin can now use

- calculator for numeric arithmetic
- current_time for current date/time
- knowledge_search for the local curriculum/reference index
- memory_lookup for session memory
- web_search for current/recent/online information when the request calls for it

Tools are conditional. A normal Grade 8 algebra lesson should usually show zero tool calls; that is correct because it can be answered from the selected curriculum and model knowledge. EduTwin should not call tools just to make the counter non-zero.

When a tool is actually useful, the bounded ReAct controller selects an allowlisted tool, validates its arguments, executes it under the existing security/budget controls, and passes the observation back to the answer workflow.

Web results are treated as untrusted source data, not instructions.

## Adaptive Multi-Agent correction

The adaptive tutoring policy is now passed to Researcher, Judge, and Writer. When the learner answers a pending check correctly, the Multi-Agent workflow is explicitly instructed to acknowledge that answer before moving to the next concept/check.

## Tests

The patch was validated with 70 passing tests and Python syntax compilation.
