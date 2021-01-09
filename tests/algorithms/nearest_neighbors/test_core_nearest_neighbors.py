import numpy as np
from algorithms.nearest_neighbors.core import classify


def test_classify():
    data = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
    labels = ["cat", "cat", "dog"]
    point = np.array([[1.0, 1.0]])
    k = 2

    assert classify(data, labels, point, k) == "cat"
