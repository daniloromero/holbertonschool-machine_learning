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
    if len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    if X.shape[0] < 2:
        raise ValueError('X must contain multiple data points')
    mean = np.sum(X, axis=0) / X.shape[0]
    X = X - mean
    conv = np.matmul(X.T, X) / (X.shape[0] - 1)

    return mean, conv
