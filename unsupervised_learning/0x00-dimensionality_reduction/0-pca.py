#!/usr/bin/env python3
"""Module that performs PCA on a data set"""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA on data set
    Args:
        X is a numpy.ndarray of shape(n, d) where
            n is the number of data points
            d is the number of dimensions in each point
        var is the fraction of the variance that the PCA to be kept
    Returns: weights matrix W: numpy.ndarray shape(d, nd) new dimensions of X
    """
    u, s, vh = np.linalg.svd(X)
    total_var = np.cumsum(s) / np.sum(s)
    r = (np.argwhere(total_var >= var))[0, 0]

    w = vh[:r + 1].T
    return w
