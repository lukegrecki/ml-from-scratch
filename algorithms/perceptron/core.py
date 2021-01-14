from typing import Tuple, Optional, List
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Model:
    bias: float
    weights: np.ndarray
    labels: Tuple[str, str] = field(default=("0", "1"))

    def output(self, point: np.ndarray) -> int:
        if np.dot(self.weights, point) + self.bias > 0:
            return 1
        else:
            return 0

    def outputs(self, points: np.ndarray) -> np.ndarray:
        return np.array([self.output(point) for point in points])

    def classify(self, points: np.ndarray) -> List[str]:
        return [self.labels[self.output(point)] for point in points]


@dataclass
class Hyperparameters:
    learning_rate: float
    initial_model: Model
    epochs: int
    tolerance: float


def train(
    data: np.ndarray,
    values: np.ndarray,
    hyperparameters: Hyperparameters,
) -> Optional[Tuple[Model, float]]:
    model = hyperparameters.initial_model

    for epoch in range(hyperparameters.epochs):
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
            return (model, error)

    return None
