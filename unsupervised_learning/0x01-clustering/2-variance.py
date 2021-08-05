#!/usr/bin/env python3
" Clustering: k-means and Gaussian Mixture Model & EM technique "

import numpy as np


def variance(X, C):
    """ calculates the total intra-cluster variance for a data set
    Arg:
        - X: np.ndarray shape (n, d) containing the data set
        - C: np.ndarray shape(k, d) with the centroid means for each cluster
    Returns: var, or None on failure
        - var: is the total variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(X.shape) != 2:
        return None

    try:
        n, d = X.shape

        dist = np.linalg.norm(X - C[:, np.newaxis], axis=2)
        min_distances = np.min(dist, axis=0)
        var = np.sum(min_distances ** 2)
        return var

    except Exception:
        return None
