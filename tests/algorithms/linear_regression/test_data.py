from algorithms.linear_regression.data import DataPoint, DataSet


def test_len():
    points = [DataPoint(0.0, 1.0), DataPoint(2.0, 3.0)]
    labels = ("X", "Y")
    ds = DataSet(points, labels)

    assert len(ds) == 2


def test_iter():
    points = [DataPoint(0.0, 1.0), DataPoint(2.0, 3.0), DataPoint(4.0, 5.0)]
    ds = DataSet(points)

    for i, point in enumerate(ds):
        assert points[i] == point


def test_get_item():
    points = [DataPoint(0.0, 1.0), DataPoint(2.0, 3.0), DataPoint(4.0, 5.0)]
    ds = DataSet(points)

    assert ds[0] == points[0]


def test_scale():
    points = [DataPoint(0.0, 1.0), DataPoint(2.0, 3.0), DataPoint(4.0, 5.0)]
    ds = DataSet(points)

    x_scaled_ds = ds.scale(0, 4.0)
    assert x_scaled_ds[2].x == 1.0

    y_scaled_ds = ds.scale(1, 5.0)
    assert y_scaled_ds[2].y == 1.0


def test_offset():
    points = [DataPoint(1.0, 1.0), DataPoint(2.0, 3.0), DataPoint(4.0, 5.0)]
    ds = DataSet(points)

    x_offset_ds = ds.offset(0, 1.0)
    assert x_offset_ds[0].x == 0.0

    y_offset_ds = ds.offset(1, 1.0)
    assert y_offset_ds[2].y == 4.0
