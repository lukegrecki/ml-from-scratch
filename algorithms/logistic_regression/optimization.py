from typing import Optional, Callable
import logging
import numpy as np
from dataclasses import dataclass
import math
from .core import Model, Solution


def log_loss(labeled_data: np.ndarray, predictions: np.ndarray) -> float:
    n = len(labeled_data)
    labels = labeled_data[:, -1]

    class_zero_loss = -labels * np.log(predictions)
    class_one_loss = (1 - labels) * np.log(1 - predictions)

    return (class_zero_loss - class_one_loss).sum() / n


@dataclass
class Hyperparameters:
    """A container class for the hyperparameters for gradient descent.

    Attributes:
        learning_rate: The learning rate of optimization.
        tolerance: The error tolerance to serve as stopping condition.
        epochs: The number of iterations to train for.
        initial_model: The logistic model to start with.
    """

    learning_rate: float
    tolerance: float
    epochs: int
    initial_model: Model


class GradientDescent:
    """The class for our gradient descent optimization.

    Args:
        hyperparameters: The hyperparameters for optimization.
        loss_function: The function to calculate loss given data and predictions.
    """

    def __init__(
        self,
        hyperparameters: Hyperparameters,
        loss_function: Callable[[np.ndarray, np.ndarray], float] = log_loss,
    ):
        self.hyperparameters = hyperparameters
        self.loss_function = loss_function
        self.model = self.hyperparameters.initial_model
        self.loss = math.inf

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

    def update(self, labeled_data: np.ndarray, predictions: np.ndarray) -> None:
        """Update the logistic model using the equations for gradient descent.

        Args:
            labeled_data: A numpy array of data points with labels.
            predictions: The class predictions given by the current logistic
                model.
        Returns:
            None.
        """
        n = len(labeled_data)

        labels = labeled_data[:, -1]
        data_points = labeled_data[:, :-1]

        normalized_data = np.insert(data_points, 0, np.ones(len(data_points)), axis=1)
        dot_product = np.dot(np.transpose(normalized_data), predictions - labels)
        gradient = self.hyperparameters.learning_rate * (dot_product / n)

        self.model = Model(
            weights=(self.model.weights - gradient), threshold=self.model.threshold
        )

    def solve(self, labeled_data: np.ndarray) -> Optional[Solution]:
        """Returns a solution to the problem if one exists.

        Args:
            labeled_data: A numpy array of classified data points.

        Returns:
            Solution if one exists, None otherwise.
        """

        self.model = self.hyperparameters.initial_model
        data_points = labeled_data[:, :-1]
        for epoch in range(self.hyperparameters.epochs):
            predictions = self.model.probability(data_points)
            self.loss = self.loss_function(labeled_data, predictions)
            self.log_progress(epoch)

            solution = self.evaluate()
            if solution:
                return solution
            self.update(labeled_data, predictions)

        return None
