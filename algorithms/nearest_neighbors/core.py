from typing import Optional, List
import numpy as np
from dataclasses import dataclass
from collections import Counter, defaultdict


@dataclass
class Neighbor:
    """A container class for nearest neighbors.

    Attributes:
        point: A numpy array representing a data point.
        distance: Distance from this point to the point being considered.
        index: Index of this data point in the original data array.
    """

    point: np.ndarray
    distance: float
    index: int


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """The standard l-2 norm of two numpy arrays.

    Args:
        a: A numpy array.
        b: A numpy array.
    """

    return np.linalg.norm(a - b)


def find_nearest_neighbors(
    data: np.ndarray, point: np.ndarray, k: int
) -> List[Neighbor]:
    """Find the k nearest neighbors given a data set and a given point.

    Args:
        data: A numpy array of data points.
        point: The given point to find the neighbors of.
        k: The number of nearest neighbors to find.

    Returns:
        A list of nearest neighbors.
    """

    nearest_neighbors = []
    max_distance = None

    for i, data_point in enumerate(data):
        d = distance(data_point, point)

        if len(nearest_neighbors) < k:
            nearest_neighbors.append(Neighbor(point=data_point, distance=d, index=i))

            if not max_distance:
                max_distance = d
            else:
                max_distance = max(max_distance, d)
        elif d < max_distance:
            farthest_neighbor_indices = [
                i for i, n in enumerate(nearest_neighbors) if n.distance == max_distance
            ]
            nearest_neighbors.pop(farthest_neighbor_indices.pop(0))
            nearest_neighbors.append(Neighbor(point=data_point, distance=d, index=i))

            if not farthest_neighbor_indices:
                max_distance = min(max_distance, d)

    return nearest_neighbors


def predict(
    data: np.ndarray, values: np.ndarray, point: np.ndarray, k: int
) -> Optional[float]:
    """Predict the value of the function at a given point.

    Args:
        data: A numpy array of data points.
        values: The given values for each data point.
        point: A numpy array of a point to predict the value of.
        k: The number of nearest neighbors to find.

    Returns:
        The weighted prediction of k nearest neighbors if exists, None otherwise.
    """

    nearest_neighbors = find_nearest_neighbors(data, point, k)
    weighted_values = [values[n.index] / n.distance for n in nearest_neighbors]

    if weighted_values:
        return sum(weighted_values)

    return None


def classify(
    data: np.ndarray,
    labels: List[str],
    point: np.ndarray,
    k: int,
    weighted: bool = False,
) -> Optional[str]:
    """Classify a point given a set of data points and labels.

    Args:
        data: A numpy array of data points.
        labels: A list of labels corresponding to the data points.
        point: The data point to classify.
        k: The number of nearest neighbors to find.
        weighted (optional): A boolean deciding if to weight the nearest classes by distance.

    Returns:
        The predicted class if it exists, None otherwise.
    """
    nearest_neighbors = find_nearest_neighbors(data, point, k)

    if not weighted:
        nearest_labels = map(lambda n: labels[n.index], nearest_neighbors)
        label_counts = Counter(nearest_labels)
        most_common_label_count = label_counts.most_common(n=1)

        if most_common_label_count:
            return most_common_label_count[0][0]
    else:
        label_weights = defaultdict(int)

        for n in nearest_neighbors:
            label_weights[labels[n.index]] += 1 / n.distance

        return max(label_weights, key=lambda l: label_weights[l])

    return None
