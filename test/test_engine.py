"""
Author: Daniel Johnson
Brief: Defines tests for the base Matrix operations, for agreement with PyTorch
"""

from testing_helpers import get_matrix_and_tensor, assert_value_equal, assert_all_grads_equal


def test_mock_layer():
    """
    Tests the forward and backward passes of a mock MLP layer
    """
    x = [[-1., 2., 3.]]
    w = [[3., 2., -1.],
         [5., -6., 4.],
         [8., 7., -9.]]
    b = [[-2., 1., -4.]]

    x_m, x_t = get_matrix_and_tensor(x)
    w_m, w_t = get_matrix_and_tensor(w)
    b_m, b_t = get_matrix_and_tensor(b)

    wt_m = w_m.transpose()
    xwt_m = x_m @ wt_m
    xwt_plus_b_m = xwt_m + b_m
    xwt_plus_b_relu_m = xwt_plus_b_m.relu()
    out_m = xwt_plus_b_relu_m.sum()

    wt_t = w_t.T
    wt_t.retain_grad()
    xwt_t = x_t @ wt_t
    xwt_t.retain_grad()
    xwt_plus_b_t = xwt_t + b_t
    xwt_plus_b_t.retain_grad()
    xwt_plus_b_relu_t = xwt_plus_b_t.relu()
    xwt_plus_b_relu_t.retain_grad()
    out_t = xwt_plus_b_relu_t.sum()
    out_t.retain_grad()

    assert_value_equal(out_m, out_t)

    out_m.backward()
    out_t.backward()

    assert_all_grads_equal([
        (out_m, out_t),
        (xwt_plus_b_relu_m, xwt_plus_b_relu_t),
        (xwt_plus_b_m, xwt_plus_b_t),
        (b_m, b_t),
        (xwt_m, xwt_t),
        (x_m, x_t),
        (wt_m, wt_t),
        (w_m, w_t),
    ])


def test_transpose():
    """
    Tests the forward and backward passes of the transpose operation
    """
    A = [[1., 2.],
         [-3., 4.],
         [5., -6.]]

    A_m, A_t = get_matrix_and_tensor(A)

    At_m = A_m.transpose()
    sum_At_m = At_m.sum()

    At_t = A_t.T
    At_t.retain_grad()
    sum_At_t = At_t.sum()
    sum_At_t.retain_grad()

    assert_value_equal(sum_At_m, sum_At_t)

    sum_At_m.backward()
    sum_At_t.backward()

    assert_all_grads_equal([
        (sum_At_m, sum_At_t),
        (At_m, At_t),
        (A_m, A_t)
    ])


def test_relu():
    """
    Tests the forward and backward passes of the relu operation
    """
    A = [[-1., 2.],
         [-3., 4.],
         [5., -6.]]

    A_m, A_t = get_matrix_and_tensor(A)

    Arelu_m = A_m.relu()
    sum_Arelu_m = Arelu_m.sum()

    Arelu_t = A_t.relu()
    Arelu_t.retain_grad()
    sum_Arelu_t = Arelu_t.sum()
    sum_Arelu_t.retain_grad()

    assert_value_equal(sum_Arelu_m, sum_Arelu_t)

    sum_Arelu_m.backward()
    sum_Arelu_t.backward()

    assert_all_grads_equal([
        (sum_Arelu_m, sum_Arelu_t),
        (Arelu_m, Arelu_t),
        (A_m, A_t)
    ])


def test_matmul():
    """
    Tests the forward and backward passes of the matmul operation
    """
    A = [[1., 2.],
         [-3., 4.]]
    B = [[2., -1.],
         [4., 3.]]

    A_m, A_t = get_matrix_and_tensor(A)
    B_m, B_t = get_matrix_and_tensor(B)

    AB_m = A_m @ B_m
    sum_AB_m = AB_m.sum()

    AB_t = A_t @ B_t
    AB_t.retain_grad()
    sum_AB_t = AB_t.sum()
    sum_AB_t.retain_grad()

    assert_value_equal(sum_AB_m, sum_AB_t)

    sum_AB_m.backward()
    sum_AB_t.backward()

    assert_all_grads_equal([
        (sum_AB_m, sum_AB_t),
        (AB_m, AB_t),
        (A_m, A_t),
        (B_m, B_t)
    ])


def test_add():
    """
    Tests the forward and backward passes of the add operation
    """
    A = [[1., 2.],
         [-3., 4.]]
    B = [[2., -1.],
         [4., 3.]]

    A_m, A_t = get_matrix_and_tensor(A)
    B_m, B_t = get_matrix_and_tensor(B)

    APlusB_m = A_m + B_m
    sum_APlusB_m = APlusB_m.sum()

    APlusB_t = A_t + B_t
    APlusB_t.retain_grad()
    sum_APlusB_t = APlusB_t.sum()
    sum_APlusB_t.retain_grad()

    assert_value_equal(sum_APlusB_m, sum_APlusB_t)

    sum_APlusB_m.backward()
    sum_APlusB_t.backward()

    assert_all_grads_equal([
        (sum_APlusB_m, sum_APlusB_t),
        (APlusB_m, APlusB_t),
        (A_m, A_t),
        (B_m, B_t)
    ])


def test_ew_mul():
    """
    Tests the forward and backward passes of the element-wise multiplication operation
    """
    A = [[1., 2.],
         [-3., 4.]]
    B = [[2., -1.],
         [4., 3.]]

    A_m, A_t = get_matrix_and_tensor(A)
    B_m, B_t = get_matrix_and_tensor(B)

    AB_m = A_m * B_m
    sum_AB_m = AB_m.sum()

    AB_t = A_t * B_t
    AB_t.retain_grad()
    sum_AB_t = AB_t.sum()
    sum_AB_t.retain_grad()

    assert_value_equal(sum_AB_m, sum_AB_t)

    sum_AB_m.backward()
    sum_AB_t.backward()

    assert_all_grads_equal([
        (sum_AB_m, sum_AB_t),
        (AB_m, AB_t),
        (A_m, A_t),
        (B_m, B_t)
    ])


def test_pow():
    """
    Tests the forward and backward passes of the pow operation
    """
    A = [[1., 2.],
         [-3., 4.]]

    A_m, A_t = get_matrix_and_tensor(A)

    A3_m = A_m ** 3
    sum_A3_m = A3_m.sum()

    A3_t = A_t ** 3
    A3_t.retain_grad()
    sum_A3_t = A3_t.sum()
    sum_A3_t.retain_grad()

    assert_value_equal(sum_A3_m, sum_A3_t)

    sum_A3_m.backward()
    sum_A3_t.backward()

    assert_all_grads_equal([
        (sum_A3_m, sum_A3_t),
        (A3_m, A3_t),
        (A_m, A_t)
    ])


def test_negation():
    """
    Tests the forward and backward passes of the negation operation
    """
    A = [[1., 2.],
         [-3., 4.]]

    A_m, A_t = get_matrix_and_tensor(A)

    neg_A_m = -A_m
    sum_neg_A_m = neg_A_m.sum()

    neg_A_t = -A_t
    neg_A_t.retain_grad()
    sum_neg_A_t = neg_A_t.sum()
    sum_neg_A_t.retain_grad()

    assert_value_equal(sum_neg_A_m, sum_neg_A_t)

    sum_neg_A_m.backward()
    sum_neg_A_t.backward()

    assert_all_grads_equal([
        (sum_neg_A_m, sum_neg_A_t),
        (neg_A_m, neg_A_t),
        (A_m, A_t)
    ])


def test_subtraction():
    """
    Tests the forward and backward passes of the subtraction operation
    """
    A = [[1., 2.],
         [-3., 4.]]
    B = [[2., -1.],
         [4., 3.]]

    A_m, A_t = get_matrix_and_tensor(A)
    B_m, B_t = get_matrix_and_tensor(B)

    AMinusB_m = A_m - B_m
    sum_AMinusB_m = AMinusB_m.sum()

    AMinusB_t = A_t - B_t
    AMinusB_t.retain_grad()
    sum_AMinusB_t = AMinusB_t.sum()
    sum_AMinusB_t.retain_grad()

    assert_value_equal(sum_AMinusB_m, sum_AMinusB_t)

    sum_AMinusB_m.backward()
    sum_AMinusB_t.backward()

    assert_all_grads_equal([
        (sum_AMinusB_m, sum_AMinusB_t),
        (AMinusB_m, AMinusB_t),
        (A_m, A_t),
        (B_m, B_t)
    ])


def test_division():
    """
    Tests the forward and backward passes of the division operation
    """
    A = [[1., 2.],
         [-3., 4.]]
    B = [[2., -1.],
         [4., 3.]]

    A_m, A_t = get_matrix_and_tensor(A)
    B_m, B_t = get_matrix_and_tensor(B)

    ADivB_m = A_m / B_m
    sum_ADivB_m = ADivB_m.sum()

    ADivB_t = A_t / B_t
    ADivB_t.retain_grad()
    sum_ADivB_t = ADivB_t.sum()
    sum_ADivB_t.retain_grad()

    assert_value_equal(sum_ADivB_m, sum_ADivB_t)

    sum_ADivB_m.backward()
    sum_ADivB_t.backward()

    assert_all_grads_equal([
        (sum_ADivB_m, sum_ADivB_t),
        (ADivB_m, ADivB_t),
        (A_m, A_t),
        (B_m, B_t)
    ])
