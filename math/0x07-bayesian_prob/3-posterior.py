#!/usr/bin/env python3
"""MOdule that calculates likelihood given vaious probabilities"""
import numpy as np


def posterior(x, n, P, Pr):
    """Calculates likelihood
    Args:
        x is the number of patients prior that develop side effects
        n is the number of patients observed
        P is 1D nump.ndarray with various hypothetical probabilities
        Pr is a 1D numpy.ndarray containing the prior beliefs of P
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
    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    total = (np.sum(Pr))
    if not (np.isclose(total, 1)):
        raise ValueError("Pr must sum to 1")
    top = (np.math.factorial(n))
    bottom = (np.math.factorial(x) * np.math.factorial(n - x))
    factorial = top / bottom
    L = factorial * (np.power(P, x)) * (np.power((1 - P), (n - x)))

    intersection = L * Pr
    marginal = np.sum(intersection)
    posterior = intersection / marginal

    return posterior
