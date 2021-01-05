from linear_regression.core import Parameters, DataSet, DataPoint
import linear_regression.core as core
import linear_regression.data as data
import pytest


def test_loss_is_zero():
    parameters = Parameters(m=2.0, b=1.0)

    data = DataSet(points=[DataPoint(1.0, 3.0), DataPoint(5.0, 11.0)])
    predictions = core.predict(data, parameters)
    assert core.loss(data, predictions) == 0.0


def test_loss_is_accurate():
    parameters = Parameters(m=2.0, b=1.0)
    data = DataSet(points=[DataPoint(1.0, 4.0), DataPoint(5.0, 12.0)])
    predictions = core.predict(data, parameters)
    assert core.loss(data, predictions) == 2.0


def test_predict():
    parameters = Parameters(m=2.0, b=1.0)
    data = DataSet(points=[DataPoint(1.0, 2.0), DataPoint(5.0, 10.0)])
    assert core.predict(data, parameters).values == [3.0, 11.0]


def test_update_is_idempotent():
    guess = Parameters(m=2.0, b=1.0)
    learning_rate = 0.1

    data = DataSet(points=[DataPoint(1.0, 3.0), DataPoint(5.0, 11.0)])
    predictions = core.predict(data, guess)
    assert core.update(guess, learning_rate, data, predictions) == guess


def test_update_is_monotonic():
    real_parameters = Parameters(m=2.0, b=1.0)
    guess = Parameters(m=3.0, b=5.0)
    learning_rate = 0.01

    data = DataSet(points=[DataPoint(1.0, 3.0), DataPoint(5.0, 11.0)])
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
    data = DataSet(points=[DataPoint(1.0, 3.0), DataPoint(5.0, 11.0)])

    solution = core.solve(learning_rate, epochs, guess, tolerance, data)
    assert core.loss(data, core.predict(data, solution)) < tolerance
