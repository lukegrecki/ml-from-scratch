import numpy as np
from algorithms.perceptron.core import Model, Hyperparameters, train


def test_model():
    bias = 1.0
    weights = np.array([2.0, 3.0])
    labels = ("cat", "dog")
    model = Model(bias, weights, labels)

    x_1 = np.array([1.0, 2.0])
    assert model.output(x_1) == 1

    x_2 = np.array([-2.0, -1.0])
    assert model.output(x_2) == 0

    points = np.array([x_1, x_2])
    assert np.array_equal(model.outputs(points), np.array([1, 0]))
    assert model.classify(np.array([x_1, x_2])) == ["dog", "cat"]


def test_train():
    data = np.array([[1.0, 1.0], [3.0, 3.0]])
    values = np.array([0, 1])
    initial_model = Model(bias=0.0, weights=np.array([0.0, 0.0]), labels=("cat", "dog"))
    hyperparameters = Hyperparameters(
        learning_rate=0.01, initial_model=initial_model, epochs=100, tolerance=0.001
    )

    learned_model, error = train(data, values, hyperparameters)

    assert learned_model.classify(data) == ["cat", "dog"]
