from typing import Optional, List
import numpy as np
import math
from dataclasses import dataclass
from collections import Counter


@dataclass
class Neighbor:
    point: np.ndarray
    distance: float
    label: str


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


def classify(
    data: np.ndarray, labels: List[str], point: np.ndarray, k: int
) -> Optional[str]:
    nearest_neighbors = []
    max_distance = math.inf

    for i, labeled_point in enumerate(data):
        d = distance(labeled_point, point)

        if len(nearest_neighbors) < k:
            nearest_neighbors.append(
                Neighbor(point=labeled_point, distance=d, label=labels[i])
            )
            max_distance = min(max_distance, d)
        elif d < max_distance:
            further_neighbor_indices = (
                i for i, n in enumerate(nearest_neighbors) if n.distance == max_distance
            )
            nearest_neighbors.pop(next(further_neighbor_indices))
            nearest_neighbors.append(Neighbor(point=point, distance=d, label=labels[i]))

    nearest_labels = map(lambda n: n.label, nearest_neighbors)
    label_counts = Counter(nearest_labels)
    most_common_label_count = label_counts.most_common(n=1)

    if most_common_label_count:
        return most_common_label_count[0][0]

    return None
