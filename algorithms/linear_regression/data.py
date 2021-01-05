import random
from dataclasses import dataclass, field
from typing import Iterable, Tuple, List, Dict, Generator
import matplotlib.pyplot as plt


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
class Parameters:
    m: float
    b: float


def plot(data: DataSet) -> None:
    x = [point.x for point in data]
    y = [point.y for point in data]

    plt.plot(x, y, "o")
