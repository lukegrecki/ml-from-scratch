from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Generator, TypeVar, Type
import logging


@dataclass
class DataPoint:
    x: float
    y: float


@dataclass
class DataSet:
    points: List[DataPoint] = field(default_factory=list)
    labels: Tuple[str, str] = ("x", "y")

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Generator[DataPoint, None, None]:
        yield from self.points

    def __getitem__(self, i: int) -> DataPoint:
        return self.points[i]

    @classmethod
    def from_dict(cls, d):
        return cls(points=list(map(lambda x: DataPoint(x, d[x]), d)))

    def add_point(self, point: DataPoint) -> None:
        self.points.append(point)

    def filter(self, f) -> List[DataPoint]:
        return list(filter(f, self.points))

    def cast(self, types):
        new_points = list(
            map(lambda p: DataPoint(types[0](p.x), types[1](p.y)), self.points)
        )
        return self.__class__(new_points)

    def scale(self, index, factor):
        new_points = []
        for point in self.points:
            if index == 0:
                new_points.append(DataPoint(x=point.x / factor, y=point.y))
            elif index == 1:
                new_points.append(DataPoint(x=point.x, y=point.y / factor))
        return self.__class__(new_points)

    def offset(self, index, number):
        new_points = []
        for point in self.points:
            if index == 0:
                new_points.append(DataPoint(x=point.x - number, y=point.y))
            elif index == 1:
                new_points.append(DataPoint(x=point.x, y=point.y - number))
        return self.__class__(new_points)


@dataclass
class Predictions:
    points: List[DataPoint] = field(default_factory=list)


@dataclass
class Parameters:
    m: float
    b: float


def loss(data: DataSet, predictions: Predictions) -> float:
    sum = 0.0
    for i, point in enumerate(data.points):
        sum += (point.y - predictions.points[i].y) ** 2

    return sum


def predict(data: DataSet, parameters: Parameters) -> Predictions:
    return Predictions(
        points=[
            DataPoint(point.x, parameters.b + parameters.m * point.x)
            for point in data.points
        ]
    )


def update(
    guess: Parameters, learning_rate: float, data: DataSet, predictions: Predictions
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
