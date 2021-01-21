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

    def probability(self, x: np.ndarray, output_class: int) -> float:
        """The probability that the given input is in the output class.

        Args:
            x: The given data point at which to evaluate the model.
            output_class: The class in question. Must be in (0, 1).

        Returns:
            The probability that x is in the output class.

        Raises:
            ValueError: If the output class is not in (0, 1).
        """

        dot_product = np.dot(self.weights, np.insert(x, 1, 0))
        probability_of_one = 1 / (1 + np.exp(-dot_product))

        if output_class == 1:
            return probability_of_one
        elif output_class == 0:
            return 1 - probability_of_one
        else:
            raise ValueError("output_class must be in (0, 1)")

    def classify(self, x: np.ndarray) -> int:
        """The predicted class for the given input.

        Args:
            x: The given data point at which to evaluate the model.

        Returns:
            The predicted class in (0, 1) of the data point.
        """

        p = self.probability(x, output_class=1)

        if p >= self.threshold:
            return 1
        else:
            return 0


@dataclass
class Solution:
    """A container class for a model with its training loss.

    Attributes:
        model: Trained logistic model.
        loss: Training loss for the linear model.
    """

    model: Model
    loss: float
