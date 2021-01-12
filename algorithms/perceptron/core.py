from typing import Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class Model:
    bias: float
    weights: np.ndarray
    label_names: Tuple[str, str] = ["0", "1"]

    def output(self, point: np.ndarray) -> int:
        if np.dot(self.weights, point) + self.bias > 0:
            return 1
        else:
            return 0

    def classify(self, point: np.ndarray) -> str:
        return self.label_names[self.output(point)]


@dataclass
class Hyperparameters:
    learning_rate: float
    initial_model: Model
    epochs: int
    tolerance: float


def train(
    data: np.ndarray,
    labels: np.ndarray,
    hyperparameters: Hyperparameters,
) -> Optional[Model]:
    model = hyperparameters.initial_model

    for epoch in hyperparameters.epochs:
        for i, point in enumerate(data):
            output = model.output(point)
            expected_output = labels[i]
            model.weights = (
                model.weights
                + hyperparameters.learning_rate * (expected_output - output) * point
            )

        outputs = np.array([model.output(point) for point in data])
        error = (1 / len(data)) * np.sum(labels - outputs)

        if error < hyperparameters.tolerance:
            return model

    return None
