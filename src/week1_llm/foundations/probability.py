from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable


def probability(count: int, total: int) -> float:
    """Return P(event) = count / total."""
    if total <= 0:
        raise ValueError("total must be greater than zero")
    if count < 0 or count > total:
        raise ValueError("count must be between 0 and total")
    return count / total


def distribution(values: Iterable[int]) -> dict[int, float]:
    """Calculate a simple empirical probability distribution."""
    values = list(values)
    if not values:
        return {}
    counts = Counter(values)
    total = len(values)
    return {key: count / total for key, count in sorted(counts.items())}


class MarkovActivityPredictor:
    """Educational first-order Markov predictor: P(next | current)."""

    def __init__(self) -> None:
        self.counts: dict[int, Counter[int]] = defaultdict(Counter)
        self.probs: dict[int, dict[int, float]] = {}

    def fit(self, sequence: Iterable[int]) -> "MarkovActivityPredictor":
        sequence = list(sequence)
        if len(sequence) < 2:
            raise ValueError("sequence must contain at least two activities")

        self.counts.clear()
        self.probs.clear()

        for current, nxt in zip(sequence, sequence[1:]):
            self.counts[current][nxt] += 1

        for current, next_counts in self.counts.items():
            total = sum(next_counts.values())
            self.probs[current] = {
                nxt: count / total for nxt, count in next_counts.items()
            }
        return self

    def conditional_probability(self, current: int, nxt: int) -> float:
        return self.probs.get(current, {}).get(nxt, 0.0)

    def predict_next_activity(self, current: int) -> int | None:
        """Return the most likely next activity; None if current is unknown."""
        next_probs = self.probs.get(current)
        if not next_probs:
            return None
        return max(next_probs, key=next_probs.get)

    def predict_distribution(self, current: int) -> dict[int, float]:
        return dict(self.probs.get(current, {}))
