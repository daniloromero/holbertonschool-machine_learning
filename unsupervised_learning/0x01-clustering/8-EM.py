#!/usr/bin/env python3
"""Module that performs Expectation maximization for GMM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Perfomrs Expectation Maximization for a GMM
    Args:
        X is a numpy.ndarray of shape (n, d) containing the data set
        k is a positive integer containing the number of clusters
        iterations positive integer containing the maximum number of iterations
         for the algorithm
        tol non-negative float containing tolerance of the log likelihood, used
            to determine early stopping i.e. if the difference is less than or
            equal to tol you should stop the algorithm
        verbose boolean that determines if you should print information about
            the algorithm
            If True, print Log Likelihood after {i} iterations: {l} every 10
                iterations and after the last iteration
            {i} is the number of iterations of the EM algorithm
            {l} is the log likelihood, rounded to 5 decimal places
    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
        pi: numpy.ndarray shape (k,) containing the priors for each cluster
        m: numpy.ndarray shape (k, d) containing the centroid means ...
        S: numpy.ndarray shape (k, d, d) containing the covariance matrices ...
        g: numpy.ndarray shape (k, n) containing the probabilities
            for each data point in each cluster
        l: log likelihood the model
    """
    if type(X) != np.ndarray or X.ndim != 2:
        return None, None, None, None, None
    if type(k) != int or k < 1:
        return None, None, None, None, None
    if type(iterations) != int or iterations < 1:
        return None, None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    lo = None
    for i in range(iterations):
        g, l = expectation(X, pi, m, S)
        if lo is not None and np.abs(l - lo) <= tol:
            if verbose:
                print('Log Likelihood after {} iterations: {}'
                      .format(i, l.round(5)))
                break
        if verbose and i % 10 == 0:
            print('Log Likelihood after {} iterations: {}'
                  .format(i, l.round(5)))
        pi, m, S = maximization(X, g)
        lo = l
    else:
        g, l = expectation(X, pi, m, S)
        if verbose:
            print('Log Likelihood after {} iterations: {}'
                  .format(iterations, l.round(5)))
    return pi, m, S, g, l
