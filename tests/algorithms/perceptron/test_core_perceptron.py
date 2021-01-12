import numpy as np
from algorithms.perceptron.core import Model, Hyperparameters, train


def test_model():
    bias = 1.0
    weights = np.array([2.0, 3.0])
    label_names = ("cat", "dog")
    model = Model(bias, weights, label_names)

    point = np.array([1.0, 2.0])
    assert model.output(point) == 1
    assert model.classify(point) == "dog"

    point = np.array([-2.0, -1.0])
    assert model.output(point) == 0
    assert model.classify(point) == "cat"
