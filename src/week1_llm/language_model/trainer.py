from __future__ import annotations

import torch
from torch import nn


def make_next_token_batch(
    token_sequences: list[list[int]],
    context_length: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn token sequences into simple next-token training examples."""
    inputs: list[list[int]] = []
    targets: list[list[int]] = []

    for seq in token_sequences:
        if len(seq) < 2:
            continue
        for start in range(0, len(seq) - 1):
            chunk = seq[start : start + context_length + 1]
            if len(chunk) < 2:
                continue

            x = chunk[:-1]
            y = chunk[1:]
            if len(x) < context_length:
                pad = [0] * (context_length - len(x))
                x = x + pad
                y = y + [-100] * len(pad)

            inputs.append(x)
            targets.append(y)

    if not inputs:
        raise ValueError("No training examples could be created.")

    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


class LocalLanguageModelTrainer:
    """Minimal Week 1 trainer. Later weeks can replace this with a full pipeline."""

    def __init__(self, model: nn.Module, learning_rate: float = 3e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def train(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        epochs: int = 10,
    ) -> list[float]:
        self.model.train()
        history: list[float] = []

        for _ in range(epochs):
            self.optimizer.zero_grad()
            _, loss, _ = self.model(inputs, targets)
            assert loss is not None
            loss.backward()
            self.optimizer.step()
            history.append(float(loss.detach().cpu()))

        return history
