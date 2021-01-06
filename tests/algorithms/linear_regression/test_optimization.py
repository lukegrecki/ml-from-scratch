import numpy as np
from algorithms.linear_regression.core import ModelParameters
from algorithms.linear_regression.optimization import GradientDescent, Hyperparameters


def test_solve_is_stationary():
    initial_model = ModelParameters(m=2.0, b=1.0)
    learning_rate = 0.1
    tolerance = 0.001
    epochs = 100
    data = np.array([[1.0, 3.0], [5.0, 11.0]])

    optimizer = GradientDescent(
        Hyperparameters(learning_rate, tolerance, epochs, initial_model)
    )
    solution = optimizer.solve(data)

    assert solution.loss == 0
    assert solution.model_parameters == initial_model


def test_solve_finds_a_correct_solution():
    true_model = ModelParameters(m=2.0, b=1.0)
    initial_model = ModelParameters(m=3.0, b=5.0)
    learning_rate = 0.01
    epochs = 1000
    tolerance = 0.001
    data = np.array([[1.0, 3.0], [5.0, 11.0]])

    optimizer = GradientDescent(
        Hyperparameters(learning_rate, tolerance, epochs, initial_model)
    )
    solution = optimizer.solve(data)

    assert solution.loss < tolerance
