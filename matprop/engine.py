"""
Author: Daniel Johnson
Brief: Matrix is defined, the primary autograd-supported type of MatProp
"""

import numpy as np


class Matrix:
    """
    A Matrix consists of a data array and a grad array.
    data represents the actual numeric values used for forward-pass calculations.
    grad represents the derivative of this "operation" w.r.t. the "output"
    Whenever an operation is performed, a graph of children is kept for backprop purposes.
    """

    def __init__(self, data, _children=(), _op=''):
        """
        :param data: 2d numpy array or 2d list, The input data of the matrix
        :param _children: tuple of Matrix, The matrices used to form this matrix
        :param _op: str, operation string for debugging purposes
        :return: A new Matrix
        """
        self.data = np.array(data).astype(np.float32)
        self.grad = np.zeros(self.data.shape).astype(np.float32)

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def backward(self):
        """
        Performs backpropagation in reverse topological order on all children
        :return: None
        """
        assert self.data.squeeze().shape == (), "Output of backward node must be scalar"
        topo_ordered_nodes = self._topo_order()

        self.grad = np.array([[1.]])
        for node in reversed(topo_ordered_nodes):
            node._backward()

    def sum(self):
        """
        ∑A = C
        :return: Matrix, result of sum operation
        """
        out = Matrix(self.data.sum(), (self,), "sum")

        def _backward():
            self.grad += out.grad  # dC/dA = ones

        out._backward = _backward
        return out

    def transpose(self):
        """
        A.T = C
        :return: Matrix, result of transpose operation
        """
        out = Matrix(self.data.T, (self,), "transpose")

        def _backward():
            self.grad += out.grad.T  # dC/dA = chain.T

        out._backward = _backward
        return out

    def relu(self):
        """
        A if A > 0 else 0 = C
        :return: Matrix, result of ReLU operation
        """
        out = Matrix(np.where(self.data > 0, self.data, 0), (self,), "ReLU")

        def _backward():
            self.grad += np.where(self.data > 0, 1, 0) * out.grad  # dC/dA = 1 if A > 0 else 0

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """
        A @ B = C
        :param other: Matrix
        :return: Matrix, result of matmul operation
        """
        other = self._force_matrix(other)
        assert self.data.shape[-1] == other.data.shape[0], \
            f"matmul shape mismatch {self.data.shape} incompatible with {other.data.shape}."
        out = Matrix(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T  # dC/dA = chain @ B.T
            other.grad += self.data.T @ out.grad  # dC/dB = A.T @ chain

        out._backward = _backward
        return out

    def __add__(self, other):
        """
        A + B = C
        :param other: Matrix
        :return: Matrix, result of add operation
        """
        other = self._force_matrix(other)
        assert self.data.shape == other.data.shape, "matrix addition needs equal shape"
        out = Matrix(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad  # dC/dA = ones
            other.grad += out.grad  # dC/dB = ones

        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        A * B = C
        :param other: Matrix
        :return: Matrix, result of element-wise multiply operation
        """
        other = self._force_matrix(other)
        assert self.data.shape == other.data.shape, "matrix elem-wise mul needs equal shape"
        out = Matrix(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad  # dC/dA = B
            other.grad += self.data * out.grad  # dC/dB = A

        out._backward = _backward
        return out

    def __pow__(self, other):
        """
        A**P = C
        :param other: int
        :return: Matrix, result of pow operation
        """
        assert isinstance(other, int), "Only integer powers supported"
        out = Matrix(self.data ** other, (self,), "pow")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad  # dC/dA = PA^(P-1)

        out._backward = _backward
        return out

    def __neg__(self):
        """
        -A = C <=> A * (-ones) = C
        :return: Matrix, result of neg operation
        """
        return self * (-1 * np.ones_like(self.data))

    def __sub__(self, other):
        """
        A - B = C <=> A + (-B) = C
        :param other: Matrix
        :return: Matrix, result of sub operation
        """
        return self + (-other)

    def __truediv__(self, other):
        """
        A / B = C <=> A * (B**-1) = C
        :param other: Matrix
        :return: Matrix, result of div operation
        """
        return self * (other ** -1)

    def __radd__(self, other):  # B + A = C
        other = self._force_matrix(other)
        return self + other

    def __rsub__(self, other):  # B - A = C
        other = self._force_matrix(other)
        return other + (-self)

    def __rmatmul__(self, other):  # B @ A = C
        other = self._force_matrix(other)
        return other @ self

    def __rmul__(self, other):  # B * A = C
        other = self._force_matrix(other)
        return self * other

    def __rtruediv__(self, other):  # B / A = C
        other = self._force_matrix(other)
        return other * (self ** -1)

    def __repr__(self):
        return f"Matrix(\ndata=\n{self.data}\ngrad=\n{self.grad}\n)"

    def _topo_order(self):
        """
        topological order all of the children in the graph
        :return: list of Matrix
        """
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        return topo

    @staticmethod
    def _force_matrix(value):
        """
        Makes sure that the input value is a Matrix, casted from numpy array if needed
        :param value: numpy array or Matrix
        :return: Matrix
        """
        assert isinstance(value, np.ndarray) or isinstance(value, Matrix), "Only Matrices Allowed"
        return value if isinstance(value, Matrix) else Matrix(value)
