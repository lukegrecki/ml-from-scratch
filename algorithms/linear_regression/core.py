import numpy as np
from typing import List, Optional, Tuple, Callable, Generator, TypeVar, Type
import logging
from .data import Parameters


def loss(data: np.ndarray, predictions: np.ndarray) -> float:
    return np.square(data[:, 1] - predictions[:, 1]).mean()


def predict(data: np.ndarray, parameters: Parameters) -> np.ndarray:
    return np.column_stack((data[:, 0], data[:, 0] * parameters.m + parameters.b))


def update(
    guess: Parameters, learning_rate: float, data: np.ndarray, predictions: np.ndarray
) -> Parameters:
    n = len(data)

    b_derivative = -2.0 * np.sum(data[:, 1] - predictions[:, 1])
    m_derivative = -2.0 * np.sum(data[:, 0] * (data[:, 1] - predictions[:, 1]))

    b = guess.b - (b_derivative / n) * learning_rate
    m = guess.m - (m_derivative / n) * learning_rate

    return Parameters(m, b)


def solve(
    learning_rate: float,
    epochs: int,
    guess: Parameters,
    tolerance: float,
    data: np.ndarray,
) -> Optional[Tuple[Parameters, float]]:
    n = len(data)

    for epoch in range(epochs):
        predictions = predict(data, guess)
        l = loss(data, predictions)

        if epoch % 500 == 0:
            logging.info(f"Training epoch {epoch}...")
            logging.info(f"Loss in current epoch is {l}")

        if l < tolerance:
            return (guess, l)

        guess = update(guess, learning_rate, data, predictions)

    return None
