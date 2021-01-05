import random
from typing import Iterable, Tuple, List, Dict
import matplotlib.pyplot as plt
from algorithms.linear_regression.core import DataSet, DataPoint, Parameters


def generate(
    number_of_points: int,
    input_range: Tuple[float, float],
    noise: float,
    parameters: Parameters,
) -> DataSet:
    data = DataSet()
    for i in range(number_of_points):
        x = random.uniform(input_range[0], input_range[1])
        y = parameters.b + parameters.m * x + random.uniform(-noise / 2, noise / 2)
        data.add_point(DataPoint(x, y))

    return data


def plot(data: DataSet) -> None:
    x = [point.x for point in data]
    y = [point.y for point in data]

    plt.plot(x, y, "o")
