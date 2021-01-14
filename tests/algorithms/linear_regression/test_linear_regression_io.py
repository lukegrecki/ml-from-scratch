import numpy as np
import algorithms.linear_regression.io as io


def test_read_csv():
    ds = io.read_csv("tests/algorithms/linear_regression/fixtures/us_population.csv")
    assert ds.shape == (60, 2)
    assert np.array_equal(ds[0, :], np.array([1960.0, 180671000.0]))


def test_to_csv():
    ds = np.array([[0.0, 1.0], [2.0, 3.0]])
    filename = "tests/algorithms/linear_regression/output/test.csv"
    labels = ("year", "population")
    io.to_csv(ds, filename, labels)

    assert np.array_equal(io.read_csv(filename), ds)
