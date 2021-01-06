import numpy as np
from dataclasses import dataclass


@dataclass
class ModelParameters:
    m: float
    b: float


@dataclass
class Solution:
    model_parameters: ModelParameters
    loss: float


def loss(data: np.ndarray, predictions: np.ndarray) -> float:
    return np.square(data[:, 1] - predictions[:, 1]).mean()


def predict(data: np.ndarray, parameters: ModelParameters) -> np.ndarray:
    return np.column_stack((data[:, 0], data[:, 0] * parameters.m + parameters.b))
