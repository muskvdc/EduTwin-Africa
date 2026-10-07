from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class WordEmbeddings:
    """
    Educational dense word-vector space.

    Supports simple Skip-gram and CBOW-style updates, cosine similarity,
    Euclidean distance, clustering and PCA visualization preparation.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.input = rng.normal(0, 0.1, (vocab_size, embedding_dim))
        self.output = rng.normal(0, 0.1, (vocab_size, embedding_dim))

    def vector(self, token_id: int) -> np.ndarray:
        return self.input[token_id]

    def cosine_similarity(self, a: int, b: int) -> float:
        va, vb = self.input[a], self.input[b]
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def euclidean_distance(self, a: int, b: int) -> float:
        return float(np.linalg.norm(self.input[a] - self.input[b]))

    def train_skipgram(
        self,
        sequences: list[list[int]],
        window_size: int = 2,
        learning_rate: float = 0.025,
        epochs: int = 5,
    ) -> None:
        """
        Simplified positive-pair Skip-gram training.

        This is an educational approximation, not a full negative-sampling
        or hierarchical-softmax implementation.
        """
        for _ in range(epochs):
            for seq in sequences:
                for center_pos, center in enumerate(seq):
                    start = max(0, center_pos - window_size)
                    end = min(len(seq), center_pos + window_size + 1)
                    for context_pos in range(start, end):
                        if context_pos == center_pos:
                            continue
                        context = seq[context_pos]
                        score = float(np.dot(self.input[center], self.output[context]))
                        error = 1.0 - score
                        center_vec = self.input[center].copy()
                        context_vec = self.output[context].copy()
                        self.input[center] += learning_rate * error * context_vec
                        self.output[context] += learning_rate * error * center_vec

    def train_cbow(
        self,
        sequences: list[list[int]],
        window_size: int = 2,
        learning_rate: float = 0.025,
        epochs: int = 5,
    ) -> None:
        """Simplified positive-pair CBOW-style training."""
        for _ in range(epochs):
            for seq in sequences:
                for center_pos, target in enumerate(seq):
                    context_ids = [
                        seq[pos]
                        for pos in range(
                            max(0, center_pos - window_size),
                            min(len(seq), center_pos + window_size + 1),
                        )
                        if pos != center_pos
                    ]
                    if not context_ids:
                        continue
                    context_vec = np.mean(self.input[context_ids], axis=0)
                    score = float(np.dot(context_vec, self.output[target]))
                    error = 1.0 - score
                    target_vec = self.output[target].copy()
                    self.output[target] += learning_rate * error * context_vec
                    self.input[context_ids] += (
                        learning_rate * error * target_vec / len(context_ids)
                    )

    def nearest_neighbors(self, token_id: int, top_k: int = 5) -> list[tuple[int, float]]:
        sims = []
        for idx in range(len(self.input)):
            if idx == token_id:
                continue
            sims.append((idx, self.cosine_similarity(token_id, idx)))
        return sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]

    def pca_2d(self, token_ids: list[int]) -> np.ndarray:
        vectors = np.array([self.input[idx] for idx in token_ids])
        return PCA(n_components=2).fit_transform(vectors)

    def kmeans(self, token_ids: list[int], n_clusters: int = 3) -> np.ndarray:
        vectors = np.array([self.input[idx] for idx in token_ids])
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(vectors)
