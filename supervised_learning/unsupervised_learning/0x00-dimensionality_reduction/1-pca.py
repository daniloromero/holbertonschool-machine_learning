#!/usr/bin/env python3
"""MOdule that performs pca on a dataset"""
import numpy as np


def pca(X, ndim):
    """Performs pca  on a dataset
    Args:
        X is  a numpy.ndarray of shape (n, d) where:
            n is the number or data points
            d is the number of dimensions in each data point
        ndim is the new dimensionality of the transform X
    Returns: T, a numpy.ndarray of shape (n, ndim) with transformed X
    """
    X_m = X - np.mean(X, axis=0)

    u, s, vh = np.linalg.svd(X_m)
    W = vh[:ndim].T

    T = np.matmul(X_m, W)

    return T
