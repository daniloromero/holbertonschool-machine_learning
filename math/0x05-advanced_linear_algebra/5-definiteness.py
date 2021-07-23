#!/usr/bin/env python3
"""Module that calculates the definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """calculates the definiteness of a matrix
    Args:
        matrix: numpy.ndarray of shape (n, n)
    Returns: the string Positive definite, Positive semi-definite, Negative
    semi-definite, Negative definite, or Indefinite, respectively
    """
    if type(matrix) is not np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')

    if len(matrix.shape) == 1:
        return None
    if np.array_equal(matrix, matrix.T):
        w, v = np.linalg.eig(matrix)
        print(w)

        if np.all(w > 0):
            return 'Positive definite'
        if np.all(w >= 0):
            return 'Positive semi-definite'
        if np.all(w < 0):
            return 'Negative definite'
        if np.all(w <= 0):
            return 'Negative semi-definite'
        else:
            return 'Indefinite'
    else:
        return None
