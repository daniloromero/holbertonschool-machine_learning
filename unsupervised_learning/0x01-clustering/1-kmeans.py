#!/usr/bin/env python3
"""Module tha performs K-means on a dataset"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset
    Args:
        X is numpy.ndarray shape(n, d) containing dataset fo clustering
            n is the number of data points
            d is the number of dimensions
        k is a positive integer giving the number of clusters
        iterations is a positive integer with max number of iterations
    Return: C, clss, or None, None on failure
        C: a numpy.ndarray of shape (k, d) with centroid means for each cluster
        clss: a numpy.ndarray of shape (n,) containing the index of the cluster
            in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) != int or k <= 0 or k > X.shape[0]:
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None
    n, d = X.shape
    # initialize random centroids
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    C = np.random.uniform(X_min, X_max, (k, d))

    for i in range(iterations + 1):
        C_clone = np.copy(C)
        # Calculate distance between points X and centroids C
        dist = np.linalg.norm(X - C[:, np.newaxis], axis=2)
        # one hot encoding matrix: datapoint to closest centroid
        clss = np.argmin(dist, axis=0)
        # Update centroids
        if (i < interations):
            for j in range(C.shape[0]):
                # reinitialize centroid If cluster contains no data points
                if X[clss == j].size == 0:
                    C[j, :] = np.random.uniform(X_min, X_max, (1, d))
                else:
                    C[j, :] = np.mean(X[clss == j], axis=0)
        # If no change in the cluster centroids the return
        if (C_clone == C).all():
            return (C, clss)

    return (C, clss)
