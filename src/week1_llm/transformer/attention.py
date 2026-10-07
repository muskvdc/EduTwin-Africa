from __future__ import annotations

import math

import torch
from torch import nn


class SelfAttention(nn.Module):
    """
    Week 1 scaled dot-product self-attention.

    Q = XWq
    K = XWk
    V = XWv
    scores = softmax(QK^T / sqrt(d_k))
    output = scores V

    A causal mask is included because this module is used by an autoregressive
    language model: a token cannot attend to future tokens.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.embedding_dim)

        seq_len = x.size(-2)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        output = weights @ v
        return self.out(output), weights
