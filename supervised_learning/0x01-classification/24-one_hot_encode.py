#!/usr/bin/env python3
""" Module that converts a numeric label vector into a one-hot matrix"""
import numpy as np


def one_hot_encode(Y, classes):
    """ converts a numeric label vector into a one-hot matrix
    Args:
        Y: is a numpy.ndarray with shape (m,) containing numeric class labels
        m: is the number of examples
        classes: is the maximum number of classes found in Y
    Returns: a one-hot encoding of Y with shape (classes, m) or None on failure
    """
    if type(classes) is not int or classes <= 0:
        return None
    if type(Y) is not np.ndarray:
        return None

    m = Y.shape[0]
    try:
        Z = np.zeros((classes, m))
        for i in range(m):
            row = Y[i]
            Z[row][i] = 1
        return Z
    except Exception:
        return None
