from typing import Tuple, Optional, List
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Model:
    """A class representing a perceptron binary classifier.

    Args:
        bias: The bias of the perceptron.
        weights: A numpy array of the perceptron weights.
        labels: A 2-tuple of strings representing the two classes.
    """

    bias: float
    weights: np.ndarray
    labels: Tuple[str, str] = field(default=("0", "1"))

    def output(self, point: np.ndarray) -> int:
        """The output of the perceptron.

        Args:
            point: The data point at which to evaluate the perceptron model.

        Returns:
            The output of the perceptron model as an integer.
        """

        if np.dot(self.weights, point) + self.bias > 0:
            return 1
        else:
            return 0

    def outputs(self, points: np.ndarray) -> np.ndarray:
        """Returns an array of outputs of the perceptron.

        Args:
            points: An array of data points which to evaluate.

        Returns:
            An array of outputs of the perceptron model.
        """

        return np.array([self.output(point) for point in points])

    def classify(self, points: np.ndarray) -> List[str]:
        """Classifies an array of points.

        Args:
            points: An array of data points which to evaluate.

        Returns:
            A list of the labels of the data points according to the model.
        """

        return [self.labels[self.output(point)] for point in points]


@dataclass
class Hyperparameters:
    """A container class for the hyperparameters for perceptron training.

    Attributes:
        learning_rate: The learning rate of optimization.
        initial_model: The perceptron model to start with.
        epochs: The number of iterations to train for.
        tolerance: The error tolerance to serve as stopping condition.
    """

    learning_rate: float
    initial_model: Model
    epochs: int
    tolerance: float


@dataclass
class Solution:
    """A container class for storing model parameters and training loss.

    Attributes:
        model: A perceptron model.
        loss: Training loss for the model.
    """

    model: Model
    loss: float


def train(
    data: np.ndarray,
    values: np.ndarray,
    hyperparameters: Hyperparameters,
) -> Optional[Solution]:
    """Trains a perceptron model given data points, labels, and hyperparameters.

    Args:
        data: An array of data points.
        values: The class values (either 0 or 1) of the data points.
        hyperparameters: The hyperparameters for optimization.

    Returns:
        A Solution if one exists, None otherwise.
    """

    model = hyperparameters.initial_model

    for _ in range(hyperparameters.epochs):
        outputs = []
        for i, point in enumerate(data):
            output = model.output(point)
            expected_output = values[i]
            model.bias = model.bias + hyperparameters.learning_rate * (
                expected_output - output
            )
            model.weights = (
                model.weights
                + hyperparameters.learning_rate * (expected_output - output) * point
            )
            outputs.append(output)

        outputs = np.array(outputs)
        error = (1 / len(data)) * np.sum(np.absolute(values - outputs))

        if error < hyperparameters.tolerance:
            return Solution(model, error)

    return None
