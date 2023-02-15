"""
Author: Daniel Johnson
Brief: Defines some helpers for repeated testing logic
"""

from matprop.engine import Matrix
from torch import Tensor
from numpy import allclose


def get_matrix_and_tensor(array: list[list[float]]) -> tuple[Matrix, Tensor]:
    m = Matrix(array)
    t = Tensor(array)
    t.requires_grad = True
    return m, t


def assert_all_grads_equal(pairs: list[tuple[Matrix, Tensor]]) -> None:
    for i, pair in enumerate(pairs):
        assert allclose(pair[0].grad, pair[1].grad.numpy()), \
            f"Gradient Disagreement at pair index {i}"


def assert_value_equal(a: Matrix, b: Tensor) -> None:
    assert allclose(a.data, b.data.item()), "Value Disagreement"
