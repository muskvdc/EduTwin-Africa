from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from week1_llm.config import config
from week1_llm.context.window import ContextWindowManager, TiktokenTokenCounter
from week1_llm.rag import RAGIndex
from week1_llm.text_processing import MarkdownAwareChunker, TextProcessor


def main() -> None:
    print("\n[1] Text processing")
    sample = "Email alex@example.com or contact @alex_dev. Call +1 555-123-4567 on 2026-09-28; budget 42."
    print("  extracted:", TextProcessor.extract_entities(sample))

    print("\n[2] Markdown-aware chunking")
    markdown = "# Attention\n\nSelf-attention mixes information across tokens.\n\n```python\nprint('hello')\n```"
    chunks = MarkdownAwareChunker(max_chars=80, overlap_chars=10).chunk(markdown)
    print("  chunks:", len(chunks))
    for chunk in chunks:
        print(f"  - chars={chunk.char_count}, structures={chunk.structure_types}")

    print("\n[3] Model-token context management")
    counter = TiktokenTokenCounter(model=config.groq_model)
    context = ContextWindowManager(
        counter,
        max_context_tokens=config.context_token_budget,
        reserved_output_tokens=config.response_token_reserve,
    )
    messages = context.fit_messages([
        {"role": "system", "content": "Explain technical ideas clearly."},
        {"role": "user", "content": "What is self-attention?"},
    ])
    print(f"  estimated input tokens: {context.count_messages(messages)} / {context.input_budget}")

    print("\n[4] RAG retrieval")
    index = RAGIndex()
    index.add_document(
        "week1-attention",
        "local://week1/attention",
        "Self-attention uses query, key, and value projections to calculate contextual token representations.",
        {"topic": "transformers"},
    )
    for result in index.search("query key value attention", top_k=2):
        print(f"  score={result.score:.3f} source={result.chunk.source}")
        print(f"  {result.chunk.text}")

    print("\nWeek 2 foundation smoke demo complete. No API call was made.")


if __name__ == "__main__":
    main()
