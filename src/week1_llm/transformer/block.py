from __future__ import annotations

import torch
from torch import nn

from week1_llm.transformer.attention import SelfAttention


class TransformerBlock(nn.Module):
    """Single educational Transformer block: attention + feed-forward + residuals."""

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = SelfAttention(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, weights = self.attention(self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, weights
