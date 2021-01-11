import numpy as np
from algorithms.nearest_neighbors.core import classify, predict, distance


def test_classify():
    data = np.array([[0.0, 0.0], [4.0, 4.0], [2.0, 2.0]])
    labels = ["cat", "dog", "cat"]

    point = np.array([[3.5, 3.5]])
    k = 1

    assert classify(data, labels, point, k) == "dog"

    point = np.array([[1.0, 1.0]])
    k = 2

    assert classify(data, labels, point, k) == "cat"


def test_weighted_classify():
    data = np.array([[0.0, 0.0], [4.0, 4.0], [2.0, 2.0]])
    labels = ["cat", "dog", "cat"]

    point = np.array([[1.0, 1.0]])
    k = 3

    assert classify(data, labels, point, k, weighted=True) == "cat"


def test_predict():
    data = np.array([[0.0, 0.0], [4.0, 4.0], [2.0, 2.0]])
    values = np.array([0.0, 1.0, 2.0])

    point = np.array([[1.0, 1.0]])
    k = 2
    expected_neighbors = [data[0], data[2]]
    expected_value = sum([n[1] / distance(point, n) for n in expected_neighbors])

    assert predict(data, values, point, k) == expected_value
