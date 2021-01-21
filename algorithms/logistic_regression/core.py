import numpy as np
from dataclasses import dataclass


@dataclass
class Model:
    """A class for storing and making predictions for a linear logistic
    model.

    Attributes:
        weights: The linear weights of the model.
    """

    weights: np.ndarray
    threshold: float = 0.5

    def probability(self, data: np.ndarray, output_class: int = 1) -> np.ndarray:
        """The probability that the given inputs are in the output class.

        Args:
            data: The given data point at which to evaluate the model.
            output_class: The class in question. Must be in (0, 1).

        Returns:
            An array of probabilites that the data points are in the output class.

        Raises:
            ValueError: If the output class is not in (0, 1).
        """
        dot_product = np.dot(
            self.weights,
            np.transpose(np.insert(data, 0, values=np.ones(len(data)), axis=1)),
        )
        probabilities_of_one = 1 / (1 + np.exp(-dot_product))

        if output_class == 1:
            return probabilities_of_one
        elif output_class == 0:
            return np.ones(len(probabilities_of_one)) - probabilities_of_one
        else:
            raise ValueError("output_class must be in (0, 1)")

    def classify(self, data: np.ndarray) -> np.ndarray:
        """The predicted classes for the given inputs.

        Args:
            data: The given data points at which to evaluate the model.

        Returns:
            The predicted classes in (0, 1) of the data points.
        """
        p = self.probability(data, output_class=1)

        return np.where(p >= self.threshold, np.ones(len(p)), np.zeros(len(p)))


@dataclass
class Solution:
    """A container class for a model with its training loss.

    Attributes:
        model: Trained logistic model.
        loss: Training loss for the linear model.
    """

    model: Model
    loss: float
