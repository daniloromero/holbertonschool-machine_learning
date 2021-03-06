#!/usr/bin/env python3
" Clustering: k-means and Gaussian Mixture Model & EM technique "

import numpy as np


def maximization(X, g):
    """ calculates the maximization step in the EM algorithm for a GMM
    Arg:
        - X: numpy.ndarray of shape (n, d) containing the data set
        - g: numpy.ndarray of shape (k, n) with the posterior probs
            for each data point in each cluster
    Returns: (pi, m, S) or (None, None, None) on failure
        - pi np.ndarray of shape (k,) containing the updated
                priors for each cluster
        - m np.ndarray of shape (k, d) containing the updated
                centroid means for each cluster
        - S np.ndarray of shape (k, d, d) containing the
                updated covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    n, d = X.shape
    probs = np.sum(g, axis=0)
    total_prob = np.ones((n, ))
    if not np.isclose(probs, total_prob).all():
        return None, None, None

    k, _ = g.shape
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        # maximization updates to mean per each cluster
        m_num = np.sum((g[i, :, np.newaxis] * X), axis=0)
        m_den = np.sum(g[i], axis=0)
        m[i] = m_num / m_den
        # maxization updates to covarianze matrix per cluster
        s_num = np.dot(g[i] * (X - m[i]).T, (X - m[i]))
        S[i] = s_num / np.sum(g[i])
        # maximization updates to prior to each cluster
        pi[i] = np.sum(g[i]) / n

    return pi, m, S
