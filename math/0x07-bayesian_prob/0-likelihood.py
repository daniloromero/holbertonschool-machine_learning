#!/usr/bin/env python3
"""MOdule that calculates likelihood given vaious probabilities"""
import numpy as np


def likelihood(x, n, P):
    """Calculates likelihood
    Args:
        x is the number of patients prior that develop side effects
        n is the number of patients observed
        P is 1D nump.ndarray with various hypothetical probabilities
    Returs: 1D numpy.ndarray with likelihoods for each probability
    """
    if type(n) is not int or n < 1:
        raise ValueError('n must be a positive integer')
    if type(x) is not int or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) != np.ndarray or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError('All values in P must be in the range [0, 1]')
    top = (np.math.factorial(n))
    bottom = (np.math.factorial(x) * np.math.factorial(n - x))
    factorial = top / bottom
    L = factorial * (np.power(P, x)) * (np.power((1 - P), (n - x)))

    return L
