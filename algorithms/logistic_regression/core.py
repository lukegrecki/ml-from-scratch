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
