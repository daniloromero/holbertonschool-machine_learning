#!/usr/bin/env python3
"""Module that finds best number of culsters for a GMM using Bayesian
Information Criterion"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """# of clusters for a GMMM using Bayesian Information Criterion
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer containing the minimum number of clusters
            to check for (inclusive)
        kmax: positive integer containing the maximum number of clusters
            to check for (inclusive)
            If kmax is None, kmax should be set to the maximum number
             of clusters possible

        iterations: positive integer containing the maximum number
            of iterations for the EM algorithm
        tol: non-negative float containing the tolerance for the EM algorithm
        verbose: boolean that determines if the EM algorithm should print
            information to the standard output
    Return: best_k, best_result, l, b, or None, None, None, None on failure
        best_k is the best value for k based on its BIC
        best_result is tuple containing pi, m, S
            pi is a numpy.ndarray of shape (k,) containing the cluster priors
                for the best number of clusters
            m is a numpy.ndarray of shape (k, d) containing the centroid means
                for the best number of clusters
            S is a numpy.ndarray of shape (k, d, d) containing the covariance
                matrices for the best number of clusters
        l is a numpy.ndarray of shape (kmax - kmin + 1) containing the log
            likelihood for each cluster size tested
        b is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
            value for each cluster size tested
    """
    n, d = X.shape
    b_lst = []
    lklhds = []
    results = []
    k_lst = []
    for k in range(kmin, kmax + 1):
        print('Entre')
        k_lst.append(k)
        pi, m, S, g, l = expectation_maximization(X, k, iterations=1000,
                                                  tol=1e-5, verbose=False)
        results.append((pi, m, S))
        # compute parameter
        p = k * d + k * d * (d + 1) / 2 + (k - 1)
        b_lst.append(p * np.log(n) - 2 * 1)
        lklhds.append(l)
    bics = np.array(b_lst)
    likelihoods = np.array(lklhds)
    best_idx = np.argmin(bics)
    best_k = k_lst[best_idx]
    best_result = results[best_idx]
    return best_k, best_result, likelihoods, bics
