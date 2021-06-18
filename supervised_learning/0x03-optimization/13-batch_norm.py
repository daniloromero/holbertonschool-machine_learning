#!/usr/bin/env python3
"""normalizes an unactivated output of a NN layer using batch normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of NN layer using batch normalization
    Args:
        Z is a numpy.ndarray shape (m, n) that should be normalized
            m is the number of data points
            n is the number of features in Z
        gamma: numpy.ndarray shape (1, n) with scales used for batch normal...
        beta: numpy.ndarray shape (1, n) wit offsets used for batch normal...
    epsilon is a small number used to avoid division by zero
    Returns: the normalized Z matrix
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_updated = gamma * Z_norm + beta
    return Z_updated
