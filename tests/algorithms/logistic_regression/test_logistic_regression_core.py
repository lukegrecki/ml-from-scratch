import pytest
import numpy as np
from algorithms.logistic_regression.core import Model


def test_probability():
    weights = np.array([1.0, 1.0, 2.0])
    data = np.array([[3.0, 5.0], [4.0, 5.0]])
    model = Model(weights)

    assert np.array_equal(
        np.around(model.probability(data, output_class=0)), np.array([0, 0])
    )
    assert np.array_equal(
        np.around(model.probability(data, output_class=1)), np.array([1, 1])
    )

    with pytest.raises(ValueError):
        model.probability(data, output_class=3)


def test_classify():
    weights = np.array([1.0, 1.0, 2.0])
    data = np.array([[3.0, 5.0], [4.0, 5.0]])
    model = Model(weights, threshold=0.75)

    assert np.array_equal(model.classify(data), np.array([1.0, 1.0]))

    model = Model(weights, threshold=0.9999999)

    assert np.array_equal(model.classify(data), np.array([0.0, 0.0]))
