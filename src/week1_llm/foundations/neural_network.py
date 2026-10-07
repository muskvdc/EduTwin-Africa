
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(
        -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    )


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(np.log(y_pred[np.arange(len(y_true)), y_true])))


class TwoLayerNeuralNetwork:
    """
    Educational dense network with an explicit forward/backward pass.

    For one output it uses sigmoid + binary cross-entropy.
    For multiple outputs it uses softmax + categorical cross-entropy.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 8,
        output_size: int = 1,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0, 0.5, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))
        self.output_size = output_size
        self.loss_history: list[float] = []

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = X @ self.W1 + self.b1
        a1 = sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        if self.output_size == 1:
            y_hat = sigmoid(z2)
        else:
            y_hat = softmax(z2)
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "y_hat": y_hat}
        return y_hat, cache

    def backward(
        self,
        cache: dict[str, np.ndarray],
        y: np.ndarray,
        learning_rate: float = 0.1,
    ) -> None:
        X = cache["X"]
        a1 = cache["a1"]
        y_hat = cache["y_hat"]
        n = len(X)

        if self.output_size == 1:
            y_binary = y.reshape(-1, 1)
            dz2 = y_hat - y_binary
        else:
            dz2 = y_hat.copy()
            dz2[np.arange(n), y.astype(int)] -= 1

        dW2 = (a1.T @ dz2) / n
        db2 = np.mean(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * a1 * (1 - a1)
        dW1 = (X.T @ dz1) / n
        db1 = np.mean(dz1, axis=0, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        learning_rate: float = 0.1,
    ) -> "TwoLayerNeuralNetwork":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.loss_history.clear()

        for _ in range(epochs):
            y_hat, cache = self.forward(X)
            if self.output_size == 1:
                loss = binary_cross_entropy(y.astype(float).reshape(-1, 1), y_hat)
            else:
                loss = categorical_cross_entropy(y.astype(int), y_hat)
            self.loss_history.append(loss)
            self.backward(cache, y, learning_rate)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y_hat, _ = self.forward(np.asarray(X, dtype=float))
        return y_hat

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)
        if self.output_size == 1:
            return (probabilities >= threshold).astype(int).ravel()
        return np.argmax(probabilities, axis=1)


class Perceptron:
    """Minimal perceptron showing weighted sum + bias + threshold."""

    def __init__(self, input_size: int, learning_rate: float = 0.1, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.1, input_size)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def predict(self, x: np.ndarray) -> int:
        return int(np.dot(x, self.weights) + self.bias >= 0)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20) -> "Perceptron":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        for _ in range(epochs):
            for x_i, y_i in zip(X, y):
                error = y_i - self.predict(x_i)
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error
        return self
