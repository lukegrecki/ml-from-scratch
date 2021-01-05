from linear_regression.core import Parameters, DataSet, DataPoint
import linear_regression.core as core
import linear_regression.data as data
import pytest


def test_loss():
    parameters = Parameters(m=2.0, b=1.0)

    data = DataSet(points=[DataPoint(1.0, 3.0), DataPoint(5.0, 11.0)])
    predictions = core.predict(data, parameters)
    assert core.loss(data, predictions) == 0.0

    data = DataSet(points=[DataPoint(1.0, 4.0), DataPoint(5.0, 12.0)])
    predictions = core.predict(data, parameters)
    assert core.loss(data, predictions) == 2.0


def test_predict():
    parameters = Parameters(m=2.0, b=1.0)
    data = DataSet(points=[DataPoint(1.0, 2.0), DataPoint(5.0, 10.0)])
    assert core.predict(data, parameters).values == [3.0, 11.0]
