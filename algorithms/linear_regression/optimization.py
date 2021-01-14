from typing import Optional
import logging
from .core import predict, loss, ModelParameters, Solution
from dataclasses import dataclass
import numpy as np
import math


@dataclass
class Hyperparameters:
    """A container class for the hyperparameters for gradient descent.

    Attributes:
        learning_rate: The learning rate of optimization.
        tolerance: The error tolerance to serve as stopping condition.
        epochs: The number of iterations to train for.
        initial_model: The linear model to start with.
    """

    learning_rate: float
    tolerance: float
    epochs: int
    initial_model: ModelParameters


class Optimizer:
    """The base class for two different forms of gradient descent.

    Args:
        hyperparameters: The hyperparameters for optimization.
    """

    def __init__(self, hyperparameters: Hyperparameters):
        self.hyperparameters = hyperparameters
        self.model = self.hyperparameters.initial_model
        self.loss = math.inf

    def update(
        self,
        data: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        """Updates the linear model using the equations for gradient descent.

        Args:
            data: A 2D numpy array of data points.
            predictions: The predictions given by the current linear model.

        Returns:
            None.
        """
        n = len(data)

        b_derivative = -2.0 * np.sum(data[:, 1] - predictions[:, 1])
        m_derivative = -2.0 * np.sum(data[:, 0] * (data[:, 1] - predictions[:, 1]))

        b = self.model.b - (b_derivative / n) * self.hyperparameters.learning_rate
        m = self.model.m - (m_derivative / n) * self.hyperparameters.learning_rate

        self.model = ModelParameters(m, b)

    def solve(self, data: np.ndarray) -> Optional[Solution]:
        raise NotImplementedError

    def log_progress(self, epoch: int) -> None:
        """Logs training progress.

        Args:
            epoch: Current iteration in training.

        Returns:
            None.
        """
        if (epoch % (self.hyperparameters.epochs // 10)) == 0:
            logging.info(f"Training epoch {epoch}...")
            logging.info(f"Loss in current epoch is {self.loss}")

    def evaluate(self) -> Optional[Solution]:
        """Evaluates the viability of the solution given the hyperparameters.

        Returns:
            Solution if the current loss is less than the tolerance, None otherwise.
        """

        if self.loss < self.hyperparameters.tolerance:
            return Solution(self.model, self.loss)
        return None


class GradientDescent(Optimizer):
    """The optimizer class for standard gradient descent.

    Args:
        hyperparameters: The hyperparameters for optimization.
    """

    def solve(self, data: np.ndarray) -> Optional[Solution]:
        """Returns a solution to the problem if one exists.

        Args:
            data: A 2D numpy array of data points.

        Returns:
            Solution if one exists, None otherwise.
        """

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
        """Returns a solution to the problem if one exists.

        Args:
            data: A 2D numpy array of data points.

        Returns:
            Solution if one exists, None otherwise.
        """

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
