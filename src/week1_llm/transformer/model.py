from __future__ import annotations

import torch
from torch import nn

from week1_llm.transformer.block import TransformerBlock


class SmallCausalLanguageModel(nn.Module):
    """
    Small Week 1 decoder-style language model foundation.

    Token IDs -> token embeddings -> positional embeddings -> Transformer block
    -> vocabulary logits.

    It is intentionally small. Later course weeks can increase depth, heads,
    context, data and training sophistication.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        context_length: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.block = TransformerBlock(embedding_dim, hidden_dim)
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        batch, seq_len = input_ids.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context length {self.context_length}."
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x, attention_weights = self.block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, attention_weights

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        do_sample: bool = True,
    ) -> torch.Tensor:
        self.eval()
        ids = input_ids.clone()

        for _ in range(max_new_tokens):
            context = ids[:, -self.context_length :]
            logits, _, _ = self(context)
            next_logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if do_sample:
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            ids = torch.cat([ids, next_token], dim=1)

        return ids
