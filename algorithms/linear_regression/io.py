import numpy as np
import csv
from typing import Tuple


def read_csv(filename: str) -> np.ndarray:
    x = []
    y = []
    with open(filename, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        labels = (header[0], header[1])
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))

    return np.column_stack((x, y))


def to_csv(data: np.ndarray, filename: str, columns: Tuple[str, str]) -> None:
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(columns))
        for point in data:
            writer.writerow([point[0], point[1]])
