import pytest
import numpy as np
from algorithms.logistic_regression.core import Model


def test_probability():
    weights = np.array([1.0, 1.0, 2.0])
    x = np.array([3.0, 5.0])
    model = Model(weights)

    assert round(model.probability(x, output_class=0), 3) == 0
    assert round(model.probability(x, output_class=1), 3) == 1

    with pytest.raises(ValueError):
        model.probability(x, output_class=3)
