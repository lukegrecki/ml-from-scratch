import numpy as np
from algorithms.linear_regression.core import ModelParameters
import algorithms.linear_regression.core as core
import pytest


def test_predict():
    parameters = ModelParameters(m=2.0, b=1.0)
    data = np.array([[1.0, 3.0], [5.0, 11.0]])
    predictions = core.predict(data, parameters)
    assert np.array_equal(predictions, np.array([[1.0, 3.0], [5.0, 11.0]]))


def test_loss_is_zero():
    parameters = ModelParameters(m=2.0, b=1.0)
    data = np.array([[1.0, 3.0], [5.0, 11.0]])
    predictions = core.predict(data, parameters)
    assert core.loss(data, predictions) == 0.0


def test_loss_is_accurate():
    parameters = ModelParameters(m=2.0, b=1.0)
    data = np.array([[1.0, 4.0], [5.0, 12.0]])
    predictions = core.predict(data, parameters)
    assert core.loss(data, predictions) == 1.0
