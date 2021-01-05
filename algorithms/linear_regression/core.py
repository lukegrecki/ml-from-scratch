from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Generator, TypeVar, Type
import csv
import logging

T = TypeVar("T", bound="DataSet")


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
    def from_dict(cls: Type[T], d) -> T:
        return cls(points=list(map(lambda x: DataPoint(x, d[x]), d)))

    @classmethod
    def from_csv(cls: Type[T], filename) -> T:
        points = []
        with open(filename, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            labels = (header[0], header[1])
            for row in reader:
                points.append(DataPoint(float(row[0]), float(row[1])))

        return cls(points, labels)

    def add_point(self, point: DataPoint) -> None:
        self.points.append(point)

    def filter(self, f) -> List[DataPoint]:
        return list(filter(f, self.points))

    def to_csv(self, filename, columns) -> None:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for point in self.points:
                writer.writerow([point.x, point.y])

    def cast(self, types) -> T:
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
