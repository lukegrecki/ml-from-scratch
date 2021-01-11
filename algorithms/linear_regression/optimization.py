from typing import Optional
import logging
from .core import predict, loss, ModelParameters, Solution
from dataclasses import dataclass
import numpy as np
import math


@dataclass
class Hyperparameters:
    learning_rate: float
    tolerance: float
    epochs: int
    initial_model: ModelParameters


class Optimizer:
    def __init__(self, hyperparameters: Hyperparameters):
        self.hyperparameters = hyperparameters
        self.model = self.hyperparameters.initial_model
        self.loss = math.inf

    def update(
        self,
        data: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        n = len(data)

        b_derivative = -2.0 * np.sum(data[:, 1] - predictions[:, 1])
        m_derivative = -2.0 * np.sum(data[:, 0] * (data[:, 1] - predictions[:, 1]))

        b = self.model.b - (b_derivative / n) * self.hyperparameters.learning_rate
        m = self.model.m - (m_derivative / n) * self.hyperparameters.learning_rate

        self.model = ModelParameters(m, b)

    def solve(self, data: np.ndarray) -> Optional[Solution]:
        raise NotImplementedError

    def log_progress(self, epoch) -> None:
        if (epoch % (self.hyperparameters.epochs // 10)) == 0:
            logging.info(f"Training epoch {epoch}...")
            logging.info(f"Loss in current epoch is {self.loss}")

    def evaluate(self) -> Optional[Solution]:
        if self.loss < self.hyperparameters.tolerance:
            return Solution(self.model, self.loss)
        return None


class GradientDescent(Optimizer):
    def solve(self, data: np.ndarray) -> Optional[Solution]:
        self.model = self.hyperparameters.initial_model
        for epoch in range(self.hyperparameters.epochs):
            predictions = predict(data, self.model)
            self.loss = loss(data, predictions)
            self.log_progress(epoch)

            solution = self.evaluate()
            if solution:
                logging.info(f"Solution found with loss {self.loss}")
                return solution
            self.update(data, predictions)

        return None


class StochasticGradientDescent(Optimizer):
    def solve(self, data: np.ndarray) -> Optional[Solution]:
        self.model = self.hyperparameters.initial_model
        indices = np.arange(len(data))
        for epoch in range(self.hyperparameters.epochs):
            np.random.shuffle(indices)

            for index in indices:
                point = data[index : index + 1]
                predictions = predict(point, self.model)
                self.update(point, predictions)

            self.loss = loss(data, predict(data, self.model))
            self.log_progress(epoch)

            solution = self.evaluate()
            if solution:
                logging.info(f"Solution found with loss {self.loss}")
                return solution

        return None
