from algorithms.linear_regression.core import DataPoint, DataSet
import algorithms.linear_regression.io as io


def test_read_csv():
    ds = io.read_csv("tests/algorithms/linear_regression/fixtures/us_population.csv")
    assert len(ds) == 60
    assert ds.labels == ("year", "population")
    assert ds.points[0] == DataPoint(1960.0, 180671000.0)


def test_to_csv():
    points = [DataPoint(0.0, 1.0), DataPoint(2.0, 3.0)]
    labels = ("X", "Y")
    ds = DataSet(points, labels)
    filename = "tests/algorithms/linear_regression/output/test.csv"
    io.to_csv(ds, filename, labels)

    assert io.read_csv(filename) == ds
