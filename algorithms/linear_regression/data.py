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
class Parameters:
    m: float
    b: float


def plot(data: DataSet) -> None:
    x = [point.x for point in data]
    y = [point.y for point in data]

    plt.plot(x, y, "o")
