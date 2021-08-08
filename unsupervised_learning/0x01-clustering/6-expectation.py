#!/usr/bin/env python3
"""Module that calculates expectation step in EM algorithm for a GMM"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates expectation step in EM algorithm for a GMM
    Args:
        X: numpy.ndarray shape (n, d) containing the data set
        pi: numpy.ndarray shape (k,) containing the priors for each cluster
        m: numpy.ndarray shape (k, d) containing centroid means for cluster
        S: numpy.ndarray shape (k, d, d) containing covariance matrices
            for each cluster
    Return: g, l, or None, None on failure
        g: numpy.ndarray shape (k, n) containing the posterior probabilities
            for each data point in each cluster
        l is the total log likelihood
    """
    if (type(X) is not np.ndarray or len(X.shape) != 2):
        return None, None

    n, d = X.shape

    if (type(pi)is not np.ndarray or len(pi.shape) != 1):
        return None, None

    k = pi.shape[0]

    if (type(m)is not np.ndarray or m.shape != (k, d)):
        return None, None

    if (type(S)is not np.ndarray or S.shape != (k, d, d)):
        return None, None

    if (not np.isclose(np.sum(pi), 1)):
        return None, None

    k = len(pi)
    n, d = X.shape
    post = np.zeros((k,n))
    for i in range (k):
        post[i] = pi[i] * pdf(X, m[i], S[i])
    den = np.sum(post, axis=0)
    post /= den
    likelihood = np.sum(np.log(den))
    return post, likelihood

    prob = num / den

    return np.maximum(prob, 1e-300)
