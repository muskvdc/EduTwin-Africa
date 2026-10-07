
from __future__ import annotations

import numpy as np

from week1_llm.digital_twin.profile import cosine_similarity


def recommend_similar_profiles(
    target_vector: np.ndarray,
    profile_vectors: dict[str, np.ndarray],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Return the most similar profiles by cosine similarity."""
    scored = [
        (name, cosine_similarity(target_vector, vector))
        for name, vector in profile_vectors.items()
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
