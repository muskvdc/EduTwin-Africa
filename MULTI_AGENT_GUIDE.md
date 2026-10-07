# Multi-Agent Mode — Learning Guide

## Try it in the browser

1. Create `.env` from `.env.example` if needed and add your rotated Groq key locally.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Start the app with `python web_app.py`.
4. Choose **Multi-Agent** in the mode selector and send a prompt.
5. After the response, inspect the activity panel and review result.

## What each module does

- `agents/orchestrator.py`: owns the workflow and the bounded retry loop.
- `agents/researcher.py`: prepares findings from the request, conversation, and optional RAG context.
- `agents/judge.py`: returns structured `pass` or `revise` feedback. Invalid review output fails closed as `revise`.
- `agents/writer.py`: turns findings into the final answer and is told to disclose unresolved uncertainty if the review limit is reached.
- `pipeline.py`: shares the existing Groq client, context manager, memory, conversation, and RAG setup with both chat modes.
- `web/app.py` and `web/static/index.html`: accept the mode selector and display the returned activity trace.

## Workflow

```text
Orchestrator
   -> Researcher
   -> Judge
      -> pass: Writer
      -> revise: feedback back to Researcher (up to AGENT_MAX_ITERATIONS)
   -> Writer
   -> final answer + trace
```

The trace is returned when the synchronous request completes; this version does
not stream each agent's status live. The trace contains stage labels and concise
review feedback, not private hidden reasoning.

## API usage and limitations

Normal Chat generally makes one Groq completion call. Multi-Agent makes at
least three calls (Researcher, Judge, Writer), and up to `2 * AGENT_MAX_ITERATIONS
+ 1` calls if every review cycle requests revision. It therefore costs more and
takes longer than Normal Chat. The same configured Groq model powers each role.

Judge approval is an automated quality check, not proof of factual truth.
Retrieval remains the project's existing local TF-IDF baseline. The browser
app is a local learning prototype; conversation state is shared in the current
single-process pipeline and should be redesigned for multi-user deployment.
