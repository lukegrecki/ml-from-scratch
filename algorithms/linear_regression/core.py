from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataPoint:
    x: float
    y: float


@dataclass
class DataSet:
    points: List[DataPoint] = field(default_factory=list)

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        yield from self.points

    def add_point(self, point: DataPoint) -> None:
        self.points.append(point)


@dataclass
class Predictions:
    values: List[float] = field(default_factory=list)


@dataclass
class Parameters:
    m: float
    b: float


def loss(data: DataSet, predictions: Predictions) -> float:
    sum = 0.0
    for i, point in enumerate(data.points):
        sum += (point.y - predictions.values[i]) ** 2

    return sum


def predict(data: DataSet, parameters: Parameters) -> Predictions:
    return Predictions(
        values=[parameters.b + parameters.m * point.x for point in data.points]
    )


def update(
    guess: Parameters, learning_rate: float, data: DataSet, predictions: Predictions
) -> Parameters:
    n = len(data)

    b_derivative = sum(
        [-2.0 * (data.points[i].y - predictions.values[i]) for i in range(n)]
    )
    m_derivative = sum(
        [
            -2.0 * data.points[i].x * (data.points[i].y - predictions.values[i])
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
) -> Optional[Parameters]:
    n = len(data)

    for epoch in range(epochs):
        predictions = predict(data, guess)
        l = loss(data, predictions)

        if l < tolerance:
            return guess

        guess = update(guess, learning_rate, data, predictions)

    return None
