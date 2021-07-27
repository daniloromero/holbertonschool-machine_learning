#!/usr/bin/env python3
"""Module that calculates a correlation matrix"""
import numpy as np


def correlation(C):
    """Calculates correlation matrix
    Args:
        C: is a numpy.ndarray of shape(d, d) containing a covariance matrix
            d id the number of dimensions
    Returns: numpy.ndarray of shape(d, d) containing the correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError('C must be a numpy.ndarray')
    h, w = np.shape(C)
    if h != w:
        raise ValueError('C must be a 2D square matrix')

    cv = C - np.mean(C, axis=0)
    cvss = (cv * cv).sum(axis=0)
    desv_x = cvss
    outer = np.outer(desv_x, desv_x)
    return np.matmul(cv.T, cv) / np.sqrt(outer)
