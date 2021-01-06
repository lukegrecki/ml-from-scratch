from typing import Optional
import logging
from .core import predict, loss, ModelParameters, Solution
from dataclasses import dataclass
import numpy as np


@dataclass
class Hyperparameters:
    learning_rate: float
    tolerance: float
    epochs: int
    initial_model: ModelParameters


class Optimizer:
    def __init__(self, hyperparameters: Hyperparameters):
        self.hyperparameters = hyperparameters
        self.solution = self.hyperparameters.initial_model

    def update(self, data: np.ndarray, predictions: np.ndarray):
        raise NotImplementedError

    def solve(self, data: np.ndarray) -> Optional[Solution]:
        for epoch in range(self.hyperparameters.epochs):
            predictions = predict(data, self.solution)
            current_loss = loss(data, predictions)

            if (epoch % (self.hyperparameters.epochs // 10)) == 0:
                logging.info(f"Training epoch {epoch}...")
                logging.info(f"Loss in current epoch is {current_loss}")

            if current_loss < self.hyperparameters.tolerance:
                return Solution(self.solution, current_loss)

            self.update(data, predictions)

        return None


class GradientDescent(Optimizer):
    def update(
        self,
        data: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        n = len(data)

        b_derivative = -2.0 * np.sum(data[:, 1] - predictions[:, 1])
        m_derivative = -2.0 * np.sum(data[:, 0] * (data[:, 1] - predictions[:, 1]))

        b = self.solution.b - (b_derivative / n) * self.hyperparameters.learning_rate
        m = self.solution.m - (m_derivative / n) * self.hyperparameters.learning_rate

        self.solution = ModelParameters(m, b)


class StochasticGradientDescent(Optimizer):
    def update(self, data: np.ndarray, predictions: np.ndarray):
        pass
