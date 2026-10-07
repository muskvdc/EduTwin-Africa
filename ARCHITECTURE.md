
# Architecture

```text
                         ┌──────────────────────┐
                         │      User / UI       │
                         │ browser or terminal  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Assistant Pipeline   │
                         │ prompt + memory +    │
                         │ conversation         │
                         └───────┬──────────────┘
                                 │
                    ┌────────────┴─────────────┐
                    │                          │
                    ▼                          ▼
          ┌─────────────────┐        ┌──────────────────┐
          │ Hosted inference│        │ Local LM         │
          │ Groq API        │        │ foundation       │
          └─────────────────┘        └────────┬─────────┘
                                              │
                                              ▼
                                  ┌──────────────────────┐
                                  │ Tokenizer            │
                                  │ tokens → IDs         │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │ Token embeddings     │
                                  │ + position vectors   │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │ Transformer block    │
                                  │ Q/K/V attention      │
                                  │ causal mask          │
                                  │ residual + LayerNorm │
                                  │ feed-forward         │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │ LM head              │
                                  │ logits over vocab    │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  next-token prediction

Learning foundations kept alongside the LM:
Probability → Markov prediction → neural network → Digital Twin
                         ↘ embeddings / similarity / PCA ↗
```

## Design principle

The architecture has two lanes:

**Learning lane:** the Week 1 mathematical/ML building blocks remain visible and testable.

**LLM lane:** tokenizer → embeddings → attention → Transformer → language-model head.

The API, prompt, memory and conversation layers sit around inference rather than pretending the hosted model is the local model.

That separation gives Weeks 2–4 a clean place to add the next capabilities without throwing away the Week 1 work.


## Week 2 extension layer

```text
Document files
    -> DocumentLoader
    -> TextProcessor / MarkdownAwareChunker
    -> RAGIndex (versioned TF-IDF baseline + source metadata)
    -> retrieved reference context
    -> AssistantPipeline

System instructions + memory + conversation + current user
    -> ContextWindowManager (tiktoken encoding + bounded turns)
    -> hosted inference provider
```

The RAG index uses immutable snapshots during search. Document changes publish a
new version and invalidate cached retrieval results. Source filtering selects
candidate rows without changing the shared corpus. The current retrieval
vectorizer is lexical TF-IDF; later course work can replace it with a stronger
embedding provider behind the retrieval boundary.


## Selectable agent collaboration

```text
Browser mode selector
   ├── Normal Chat -> existing AssistantPipeline -> Groq
   └── Multi-Agent -> AssistantPipeline context/RAG
                       -> bounded ReAct Tool Use
                          -> Thought -> Action -> Observation -> Answer
                       -> Orchestrator
                          -> Researcher -> Judge
                               ^           |
                               |-- feedback/retry (bounded)
                          -> Writer -> final answer + trace
```

All agents share the existing Groq client/model configuration. The
Researcher receives the bounded conversation and optional retrieved context;
the Judge returns a structured pass/revise decision; the Orchestrator feeds
revision feedback back to the Researcher up to the configured limit; the
Writer produces the final response. If the limit is reached without approval,
the API and UI report that state rather than implying approval.


## Week 3 tool-use boundary

ReAct is implemented as a bounded tool-use controller rather than a fifth
application agent. The four-agent collaboration remains:

```text
Orchestrator -> Researcher -> Judge -> Writer
```

Before that collaboration, the ReAct controller can select an allowlisted tool,
validate its schema-defined arguments through `ToolRegistry`, execute the
trusted handler, feed the result back as an observation, and finish with a
concise tool-use conclusion.

Each tool has an explicit model-facing contract:

```text
name
description
parameters (JSON Schema)
```

The registry remains the enforcement point for allowlisting, validation,
approval, budgets and serializable outputs.
