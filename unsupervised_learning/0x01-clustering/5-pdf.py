#!/usr/bin/env python3
"""Module: calculates probability density function of Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """ calculates the probability density function of a Gaussian distri
    Arg:
        - X: np.ndarray of shape (n, d) containing the data points
                whose PDF should be evaluated
        - m: np.ndarray of shape (d,) with the mean of the distribution
        - S: np.ndarray of shape (d, d) with the covariance of the distri
    Returns: (P), or (None) on failure
        - P: np.ndarray shape (n,) with the PDF values for each data point
                All values in P should have a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1] or X.shape[1] != S.shape[1]:
        return None
    Xm = X - m
    n, d = X.shape
    e = -0.5 * np.sum(Xm * np.matmul(np.linalg.inv(S), Xm.T).T, axis=1)
    num = np.exp(e)
    det = np.linalg.det(S)
    den = np.sqrt(((2 * np.pi) ** d) * det)

    prob = num / den

    return np.maximum(prob, 1e-300)
