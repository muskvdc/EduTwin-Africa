
from __future__ import annotations

import numpy as np

from week1_llm.foundations.neural_network import TwoLayerNeuralNetwork


class DigitalTwinPredictor:
    """
    Week 1 multi-class Digital Twin predictor.

    Input features represent a simplified current user/day state.
    Output classes represent the predicted next activity.
    """

    def __init__(
        self,
        input_size: int,
        num_activities: int = 5,
        hidden_size: int = 16,
        seed: int = 42,
    ):
        self.network = TwoLayerNeuralNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=num_activities,
            seed=seed,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        learning_rate: float = 0.1,
    ) -> "DigitalTwinPredictor":
        self.network.fit(X, y, epochs, learning_rate)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.network.predict(X)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        return self.network.predict_proba(X)

    @property
    def loss_history(self) -> list[float]:
        return self.network.loss_history
