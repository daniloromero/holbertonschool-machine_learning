#!/usr/bin/env python3
"""Module to normalize a matrix"""
import numpy as np


def normalize(X, m, s):
    """ normalization of a matrix
    Args:
        X: is the numpy.ndarray of shape (d, nx) to normalize
            d is the number of data points
            nx is the number of features
        m: numpy.ndarray of shape (nx,) contains mean of all features of X
        s: numpy.ndarray of shape (nx,) contains std deviation X's features
        Returns: The normalized X matrix
    """
    X -= m
    X /= s
    return X
