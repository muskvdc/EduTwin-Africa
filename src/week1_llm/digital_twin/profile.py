
from __future__ import annotations

import numpy as np


def profile_to_vector(features: list[float], dimension: int = 32, seed: int = 42) -> np.ndarray:
    """
    Turn a compact numerical profile into a fixed-size vector.

    The projection is deterministic and educational; it is not a learned
    production user embedding.
    """
    rng = np.random.default_rng(seed)
    projection = rng.normal(0, 1 / np.sqrt(max(1, len(features))), (len(features), dimension))
    vector = np.asarray(features, dtype=float) @ projection
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))
