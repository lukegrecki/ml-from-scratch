import numpy as np
from algorithms.linear_regression.core import Parameters
import algorithms.linear_regression.core as core
import algorithms.linear_regression.data as data
import pytest


def test_predict():
    parameters = Parameters(m=2.0, b=1.0)
    data = np.array([[1.0, 3.0], [5.0, 11.0]])
    predictions = core.predict(data, parameters)
    assert np.array_equal(predictions, np.array([[1.0, 3.0], [5.0, 11.0]]))


def test_loss_is_zero():
    parameters = Parameters(m=2.0, b=1.0)
    data = np.array([[1.0, 3.0], [5.0, 11.0]])
    predictions = core.predict(data, parameters)
    assert core.loss(data, predictions) == 0.0


def test_loss_is_accurate():
    parameters = Parameters(m=2.0, b=1.0)
    data = np.array([[1.0, 4.0], [5.0, 12.0]])
    predictions = core.predict(data, parameters)
    assert core.loss(data, predictions) == 1.0


def test_update_is_idempotent():
    guess = Parameters(m=2.0, b=1.0)
    learning_rate = 0.1
    data = np.array([[1.0, 3.0], [5.0, 11.0]])
    predictions = core.predict(data, guess)
    assert core.update(guess, learning_rate, data, predictions) == guess


def test_update_is_monotonic():
    real_parameters = Parameters(m=2.0, b=1.0)
    guess = Parameters(m=3.0, b=5.0)
    learning_rate = 0.01
    data = np.array([[1.0, 3.0], [5.0, 11.0]])

    predictions = core.predict(data, guess)
    updated_guess = core.update(guess, learning_rate, data, predictions)
    updated_predictions = core.predict(data, updated_guess)

    assert core.loss(data, predictions) > core.loss(data, updated_predictions)


def test_solve_finds_a_correct_solution():
    real_parameters = Parameters(m=2.0, b=1.0)
    guess = Parameters(m=3.0, b=5.0)
    learning_rate = 0.01
    epochs = 1000
    tolerance = 0.001
    data = np.array([[1.0, 3.0], [5.0, 11.0]])

    solution, loss = core.solve(learning_rate, epochs, guess, tolerance, data)
    assert loss < tolerance
