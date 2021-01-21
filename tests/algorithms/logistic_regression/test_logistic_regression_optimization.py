import numpy as np
from algorithms.logistic_regression.core import Model
from algorithms.logistic_regression.optimization import Hyperparameters, GradientDescent


def test_solve():
    weights = np.array([1.0, 1.0, 2.0])
    labeled_data = np.array([[3.0, 5.0, 1.0], [4.0, 5.0, 1.0]])
    initial_model = Model(weights=weights, threshold=0.75)
    hyperparameters = Hyperparameters(
        learning_rate=0.01, tolerance=0.001, epochs=1000, initial_model=initial_model
    )

    optimizer = GradientDescent(hyperparameters)
    solution = optimizer.solve(labeled_data)

    assert solution
    assert solution.loss < hyperparameters.tolerance
