
# Week 1 foundation + Week 2 integration status

## Implemented in this foundation

### Probability & sequential prediction
- empirical probability distributions
- conditional probability
- first-order Markov activity prediction

### Neural networks
- perceptron
- sigmoid
- hidden layer
- forward pass
- backpropagation
- loss
- binary and multi-class output
- Digital Twin next-activity predictor
- 1,000 synthetic behavioral samples with sleep/mood/energy/day/time features
- 80/20 train/test split, softmax multiclass prediction, scenario testing and evaluation

### Tokenization & vocabulary
- tokens
- token IDs
- vocabulary construction
- `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`
- unknown-token handling

### Embeddings
- dense vectors
- Skip-gram-style training
- CBOW-style training
- cosine similarity
- Euclidean distance
- nearest-neighbor lookup
- PCA / t-SNE visualization
- 32-dimensional educational embedding space
- user/profile vector foundation
- clustering foundation
- similarity-based profile recommendations

### LLM architecture
- token embedding
- positional embedding
- Q/K/V
- scaled dot-product attention
- softmax attention weights
- causal masking
- residual connections
- LayerNorm
- feed-forward network
- language-model head
- next-token prediction
- tiny autoregressive generation

### Prompting
- system instructions
- zero-shot
- one-shot
- few-shot message structure

### API integration
- base URL
- API key from environment
- auth header
- model selection
- messages
- temperature
- single-turn calls
- multi-turn conversation wrapper

### Application foundation
- bounded conversation history
- simple persistent memory/profile
- browser chat
- FastAPI backend
- local model kept separate from hosted inference

## Week 2 upgrades now integrated

- text normalization and mutually exclusive email/handle/date/phone/number extraction
- Markdown-aware chunking with parameter validation and atomic tables/fenced code blocks
- filesystem loader for text formats and text-based PDFs, with content sniffing and file-size limits
- model-token-aware context budgeting through tiktoken, with protected system/latest user messages
- optional retrieved context that can be dropped before core instructions when the budget is tight
- versioned TF-IDF RAG index, immutable search snapshots, safe source filters, cache invalidation on document changes
- optional RAG context connection in the assistant pipeline; files in `data/knowledge/` are reindexed at app startup

The retrieval index is in-memory and rebuilt from source files at startup. The retrieval vectorizer is still an educational lexical baseline; semantic embedding retrieval and richer tool/agent capabilities remain future course work. A selectable four-agent baseline is now included.

## Deliberately not completed yet

Week 2–4 are supposed to determine these:
- production-grade BPE/subword tokenization
- richer datasets and data pipeline
- proper train/validation/test evaluation
- scalable training
- stronger Transformer architecture
- multi-head attention if taught
- improved positional representation if taught
- checkpointing
- serious inference/sampling
- final agent/application capabilities
- final course requirements

**Handoff state:** Week 1 foundation preserved with tested Week 2 processing/context/RAG upgrades. Weeks 3–4 should extend these interfaces rather than replace the existing model foundation.


## Selectable multi-agent extension

The project now includes an optional four-agent workflow that reuses the existing
Groq client and Week 2 context/RAG pipeline. Normal Chat remains the default.
Multi-Agent Mode coordinates Researcher → Judge (bounded feedback loop) → Writer,
and returns a concise activity trace with review status and iteration count.
The UI displays this trace after the request completes; it is not live streaming.
