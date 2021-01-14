import numpy as np
from dataclasses import dataclass


@dataclass
class ModelParameters:
    """A container class for storing model parameters for a 2D linear model.

    Attributes:
        m: The slope of the line.
        b: The y-intercept of the line.
    """

    m: float
    b: float


@dataclass
class Solution:
    """A container class for storing model parameters and training loss.

    Attributes:
        model_parameters: 2D linear model parameters.
        loss: Training loss for the linear model.
    """

    model_parameters: ModelParameters
    loss: float


def loss(data: np.ndarray, predictions: np.ndarray) -> float:
    """Calculcates mean squared error between data and predictions.

    Args:
        data: 2D numpy array of given data points.
        predictions: 2D numpy array of predictions.

    Returns:
        The mean squared error between the two arrays.
    """

    return np.square(data[:, 1] - predictions[:, 1]).mean()


def predict(data: np.ndarray, parameters: ModelParameters) -> np.ndarray:
    """Returns predictions of a 2D linear model.

    Args:
        data: 2D numpy array of given data points.
        parameters: Parameters of a 2D linear model.

    Returns:
        A 2D numpy array of the given inputs and the linear predictions.
    """

    return np.column_stack((data[:, 0], data[:, 0] * parameters.m + parameters.b))
