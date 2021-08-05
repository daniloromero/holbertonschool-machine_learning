#!/usr/bin/env python3
"""Module that initializes cluster centroids for k-means"""
import numpy as np


def initialize(X, k):
    """Initializes clusters for k-means
    Args:
        X is numpy.ndarray shape(n, d) containing dataset fo clustering
            n is the number of data points
            d is the number of dimensions
        k is a positive integer giving the number of clusters
    Returs: numpy.ndarray shape(k, d) with initializes centroids, None if fails
    """
    n, d = X.shape
    return np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (k, d))