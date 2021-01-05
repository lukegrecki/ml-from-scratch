import csv
from typing import Tuple
from .core import DataPoint, DataSet


def read_csv(filename: str) -> DataSet:
    points = []
    with open(filename, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        labels = (header[0], header[1])
        for row in reader:
            points.append(DataPoint(float(row[0]), float(row[1])))

    return DataSet(points, labels)


def to_csv(ds: DataSet, filename: str, columns: Tuple[str, str]) -> None:
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for point in ds.points:
            writer.writerow([point.x, point.y])
