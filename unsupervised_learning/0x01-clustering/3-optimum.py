#!/usr/bin/env python3
"""Module tha tests for the optimum number of clusters by variance"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ Test for the optimum number of clusters by variance
    Arg:
        X: np.ndarray of shape (n, d) containing the data set
        kmin: positive int, minimum number of clusters to check
                for (inclusive)
        kmax: positive int, maximum number of clusters to check
        iterations: is a positive integer containing the maximum
                    number of iterations for K-means
    Returns: (results, d_vars) or (None, None) on failure
        - results: list, outputs of K-means for each cluster size
        - d_vars: list, with the difference in varianca from smallest
                    cluster size for each cluster size
    """
    if (type(X) is not np.ndarray or len(X.shape) != 2):
        return None, None

    if (type(kmin) is not int or kmin < 1):
        return None, None

    if (kmax is None):
        kmax = X.shape[0]
    elif (type(kmax) is not int or kmax <= kmin):
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None
    try:
        results = []
        d_vars = []
        for k in range(kmin, kmax + 1):
            C_k, clss = kmeans(X, k, iterations)
            kmin_var = variance(X, C_k)
            if k == kmin:
                first_var = kmin_var
            results.append((C_k, clss))
            d_vars.append(first_var - kmin_var)
        return results, d_vars
    except Exception:
        return None, None
