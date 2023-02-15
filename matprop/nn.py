"""
Author: Daniel Johnson
Brief: Neural network modules are defined using Matrix for autograd
"""

import numpy as np
from matprop.engine import Matrix


class Module:
    """
    Parent type for all network parts
    """

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def parameters(self):
        return []


class FCLayer(Module):
    """
    A fully-connected layer with the option of ReLU activation
    """

    def __init__(self, n_inputs, n_outputs, nonlin=True):
        """

        :param n_inputs: int, number of input features coming into the layer
        :param n_outputs: int, number of outputs going out of the layer
        :param nonlin: bool, whether or not to apply ReLU activation
        """
        self.weights = Matrix(np.random.uniform(-1, 1, size=(n_outputs, n_inputs)))
        self.biases = Matrix(np.random.uniform(-1, 1, size=(1, n_outputs)))
        self.nonlin = nonlin

    def __call__(self, x):
        """
        xW.T + b (the __rmatmul__ is because of numpy compatability problems, don't judge me)
        :param x: Matrix or numpy array
        :return: Matrix, the layer output
        """
        z = self.weights.transpose().__rmatmul__(x) + self.biases
        return z.relu() if self.nonlin else z

    def parameters(self):
        """
        :return: list of Matrix, the layer's parameters
        """
        return [self.weights, self.biases]


class MLP(Module):
    """
    A Collection of fully connected layers, making up a M(ulti-)L(ayer-)P(erceptron)
    """

    def __init__(self, n_inputs, layer_outputs):
        """
        :param n_inputs: int, number of input features coming into the network
        :param layer_outputs: list of int, number of outputs making up the rest of the network
        """
        sizes = [n_inputs] + layer_outputs
        self.layers = [FCLayer(sizes[i], sizes[i + 1], nonlin=(i != len(layer_outputs) - 1))
                       for i in range(len(layer_outputs))]  # nonlin for non-output layers

    def __call__(self, x):
        """
        :param x: Matrix or numpy array
        :return: Matrix, the MLP output
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        :return: list of Matrix, every parameter of every layer in the network
        """
        return [p for layer in self.layers for p in layer.parameters()]
