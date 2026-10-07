# Data

## Curriculum knowledge base

EduTwin Africa reads curriculum material from `data/knowledge/*.txt`.

Each curriculum file should identify its scope near the top, for example:

```text
GRADE 8 MATHEMATICS
SUBJECT: MATHEMATICS
TOPIC: ALGEBRA
```

The Streamlit UI discovers Grade → Subject → Topic options from these files. A
learning context is only enabled when a matching curriculum source exists, so the
application does not invent unsupported curriculum coverage.

Keep curriculum-specific material in the knowledge files and use the RAG layer to
retrieve the relevant excerpts.
