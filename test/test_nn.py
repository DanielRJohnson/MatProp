"""
Author: Daniel Johnson
Brief: Defines tests for FCLayer and MLP passes for agreement with PyTorch
"""

from matprop.nn import MLP, FCLayer
from matprop.engine import Matrix
from matprop.losses import mse_loss
import torch

from testing_helpers import assert_value_equal, assert_all_grads_equal


def test_fclayer():
    """
    Tests the forward pass and backward pass of one FCLayer
    """
    X_m = [Matrix([[1., 4.]]),
           Matrix([[2., -1]]),
           Matrix([[3., 9.]])]
    y_m = [Matrix([[5.]]),
           Matrix([[1.]]),
           Matrix([[12.]])]

    X_t = torch.Tensor([[1., 4.],
                        [2., -1],
                        [3., 9.]])
    y_t = torch.Tensor([[5.],
                        [1.],
                        [12.]])

    lin_m = FCLayer(2, 1)
    lin_m.weights = Matrix([[-2., 1.]])
    lin_m.biases = Matrix([[3.]])

    lin_t = torch.nn.Linear(2, 1)
    lin_t.weight = torch.nn.Parameter(torch.tensor([[-2., 1.]]))
    lin_t.bias = torch.nn.Parameter(torch.tensor([3.]))

    preds_m = [lin_m(x) for x in X_m]
    preds_t = lin_t(X_t).relu()

    loss_m = mse_loss(preds_m, y_m)
    lossfunc_t = torch.nn.MSELoss()
    loss_t = lossfunc_t(preds_t, y_t)
    loss_t.retain_grad()

    assert_value_equal(loss_m, loss_t)

    loss_m.backward()
    loss_t.backward()

    assert_all_grads_equal([
        (loss_m, loss_t),
        (lin_m.weights, lin_t.weight),
        (lin_m.biases, lin_t.bias),
    ])


def test_mlp():
    """
    Tests the forward and backward pass of an MLP with multiple layers
    """
    X_m = [Matrix([[1., 4.]]),
           Matrix([[2., -1]]),
           Matrix([[3., 9.]])]
    y_m = [Matrix([[5.]]),
           Matrix([[1.]]),
           Matrix([[12.]])]

    X_t = torch.Tensor([[1., 4.],
                        [2., -1],
                        [3., 9.]])
    y_t = torch.Tensor([[5.],
                        [1.],
                        [12.]])

    l1_w = [[1., 2.],
            [-1., 2.],
            [3., 4.]]
    l1_b = [[3., 2., 1.]]

    l2_w = [[4., 3., -2.]]
    l2_b = [[1.]]

    mlp_m = MLP(n_inputs=2, layer_outputs=[3, 1])

    mlp_m.layers[0].weights = Matrix(l1_w)
    mlp_m.layers[0].biases = Matrix(l1_b)
    mlp_m.layers[1].weights = Matrix(l2_w)
    mlp_m.layers[1].biases = Matrix(l2_b)

    mlp_t = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 1)
    )

    mlp_t._modules["0"].weight = torch.nn.Parameter(torch.Tensor(l1_w))
    mlp_t._modules["0"].bias = torch.nn.Parameter(torch.Tensor(l1_b))
    mlp_t._modules["2"].weight = torch.nn.Parameter(torch.Tensor(l2_w))
    mlp_t._modules["2"].bias = torch.nn.Parameter(torch.Tensor(l2_b))

    preds_m = [mlp_m(x) for x in X_m]
    preds_t = mlp_t(X_t)

    loss_m = mse_loss(preds_m, y_m)
    lossfunc_t = torch.nn.MSELoss()
    loss_t = lossfunc_t(preds_t, y_t)
    loss_t.retain_grad()

    assert_value_equal(loss_m, loss_t)

    loss_m.backward()
    loss_t.backward()

    assert_all_grads_equal([
        (loss_m, loss_t),
        (mlp_m.layers[0].weights, mlp_t._modules["0"].weight),
        (mlp_m.layers[0].biases, mlp_t._modules["0"].bias),
        (mlp_m.layers[1].weights, mlp_t._modules["2"].weight),
        (mlp_m.layers[1].biases, mlp_t._modules["2"].bias),
    ])
