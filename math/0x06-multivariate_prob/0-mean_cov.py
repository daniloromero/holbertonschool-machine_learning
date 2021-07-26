#!/usr/bin/env python3
"""Module that calculates the mean and covariance of a data set"""
import numpy as np


def mean_cov(X):
    """Calculates mean and covariance of data set X
    Args:
        X: numpy.ndarray of shape(n, d) containing the data set:
            n is the number of data points
            d is the number of dimensions in each data point
    Returns: mean, conv
        mean is a numpy.ndarray of shape(1, d) containing the mean
        cov is a numpy.ndarray of shape(d, d) containing covariance matrix
    """
    if type(X) is not np.ndarray and len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    n, d = np.shape(X)
    if n < 2:
        raise ValueError('X must contain multiple data points')
    mean = np.mean(X, axis=0).reshape(1, d)
    X = X - mean
    cov = np.matmul(X.T, X) / (n - 1)

    return mean, cov
