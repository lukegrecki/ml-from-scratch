from typing import List, Optional, Tuple, Callable, Generator, TypeVar, Type
import logging
from .data import DataPoint, DataSet, Parameters


def loss(data: DataSet, predictions: DataSet) -> float:
    sum = 0.0
    for i, point in enumerate(data.points):
        sum += (point.y - predictions.points[i].y) ** 2

    return sum


def predict(data: DataSet, parameters: Parameters) -> DataSet:
    return DataSet(
        points=[
            DataPoint(point.x, parameters.b + parameters.m * point.x)
            for point in data.points
        ]
    )


def update(
    guess: Parameters, learning_rate: float, data: DataSet, predictions: DataSet
) -> Parameters:
    n = len(data)

    b_derivative = sum(
        [-2.0 * (data.points[i].y - predictions.points[i].y) for i in range(n)]
    )
    m_derivative = sum(
        [
            -2.0 * data.points[i].x * (data.points[i].y - predictions.points[i].y)
            for i in range(n)
        ]
    )

    b = guess.b - (b_derivative / n) * learning_rate
    m = guess.m - (m_derivative / n) * learning_rate

    return Parameters(m, b)


def solve(
    learning_rate: float,
    epochs: int,
    guess: Parameters,
    tolerance: float,
    data: DataSet,
) -> Optional[Tuple[Parameters, float]]:
    n = len(data)

    for epoch in range(epochs):
        logging.info(f"Training epoch {epoch}...")
        predictions = predict(data, guess)
        l = loss(data, predictions)

        logging.info(f"Loss in current epoch is {l}")
        if l < tolerance:
            return (guess, l)

        guess = update(guess, learning_rate, data, predictions)

    return None
