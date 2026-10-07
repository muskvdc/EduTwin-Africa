
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_to_2d(vectors: np.ndarray, method: str = "pca") -> np.ndarray:
    if method.lower() == "pca":
        return PCA(n_components=2).fit_transform(vectors)
    if method.lower() == "tsne":
        if len(vectors) < 3:
            raise ValueError("t-SNE needs at least three vectors.")
        return TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)-1)).fit_transform(vectors)
    raise ValueError("method must be 'pca' or 'tsne'")


def plot_embeddings(
    words: list[str],
    vectors: np.ndarray,
    method: str = "pca",
    max_labels: int = 30,
):
    coords = reduce_to_2d(vectors, method)
    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
    for i, word in enumerate(words[:max_labels]):
        plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=9)
    plt.title(f"Word Embeddings Visualization ({method.upper()})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.show()
    return coords


def plot_attention(words: list[str], scores: np.ndarray):
    """Display the attention-score matrix as a Week 1 heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(scores, aspect="auto")
    plt.xticks(range(len(words)), words, rotation=45, ha="right")
    plt.yticks(range(len(words)), words)
    plt.title("Attention Weights")
    plt.xlabel("Key / attended token")
    plt.ylabel("Query token")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
