from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from week1_llm.language_model.trainer import LocalLanguageModelTrainer, make_next_token_batch
from week1_llm.tokenizer.tokenizer import SimpleTokenizer
from week1_llm.transformer.model import SmallCausalLanguageModel


def main() -> None:
    texts = [
        "I am learning artificial intelligence.",
        "I am learning how language models predict tokens.",
        "Transformers use attention to process context.",
        "Attention connects tokens through learned relationships.",
        "A language model predicts the next token.",
    ]

    tokenizer = SimpleTokenizer(vocab_size=128)
    tokenizer.build_vocab(texts)
    sequences = [tokenizer.encode(text) for text in texts]

    context_length = 16
    inputs, targets = make_next_token_batch(sequences, context_length)

    model = SmallCausalLanguageModel(
        vocab_size=tokenizer.vocab_size_actual,
        embedding_dim=32,
        context_length=context_length,
        hidden_dim=64,
    )
    trainer = LocalLanguageModelTrainer(model, learning_rate=3e-3)
    history = trainer.train(inputs, targets, epochs=20)

    print("Training complete.")
    print(f"Vocabulary: {tokenizer.vocab_size_actual}")
    print(f"Final loss: {history[-1]:.4f}")

    prompt = "a language model"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=8,
        temperature=0.8,
        do_sample=False,
    )
    print("Generated:", tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
