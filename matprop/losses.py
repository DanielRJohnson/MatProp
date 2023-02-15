"""
Author: Daniel Johnson
Brief: Defines loss functions
"""

from matprop.engine import Matrix


def mse_loss(preds, truths):
    """
    Mean Squared Error Loss: (1/N)*∑(pred - truth)**2
    :param preds: list of Matrix, predictions
    :param truths: list of Matrix or numpy array, ground truths
    :return: Matrix, the total MSE loss
    """
    errors = [pred - truth for truth, pred in zip(truths, preds)]
    sq_errors = [error ** 2 for error in errors]
    sum_sq_errors = sum(sq_errors, start=Matrix([[0.]]))
    return sum_sq_errors / Matrix([[len(preds)]])
