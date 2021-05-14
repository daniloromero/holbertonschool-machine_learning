#!/usr/bin/env python3
"""Module that converts a one-hot matrix into a vector of labels"""
import numpy as np


def one_hot_decode(one_hot):
    """ converts a one-hot matrix into a vector of labels
    Args:
        one_hot: a one-hot encoded numpy.ndarray with shape (classes, m)
            classes is the maximum number of classes
            m is the number of examples
        Returns: a numpy.ndarray with shape (m, ) with numeric labels for
        each example, or None on failure
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    if np.any((one_hot != 0) & (one_hot != 1)):
        return None
    _, m = one_hot.shape
    decoded = np.zeros(m, dtype=int)
    for index, array in enumerate(one_hot):
        position = np.where(array == 1)
        decoded[position] = index
    return decoded
