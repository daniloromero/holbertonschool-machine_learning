#!/usr/bin/env python3
""" Module that converts a numeric label vector into a one-hot matrix"""
import numpy as np


def one_hot_encode(Y, classes):
    """ converts a numeric label vector into a one-hot matrix
    Args:
        Y: is a numpy.ndarray with shape (m,) containing numeric class labels
        m: is the number of examples
        classes: is the maximum number of classes found in Y
    Returns: a one-hot encoding of Y with shape (classes, m), or None on failure
    """
    if type(classes) is not int or classes <= 0:
        return None
    n = len(Y)
    try:
        Z = np.zeros((classes, (n + 1)))
        for i in range(len(Y)):
            row = Y[i]
            Z[row][i] = 1
        return Z
    except Exception:
        return None
