#!/usr/bin/env python3
"""Module that calculates a GMM form a data set"""
import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM form a dataset
    Args:
        X is a numpy.ndarray of shape (n, d) containing the dataset
        k is the number of clusters
    Return:pi, m, S, clss, bic
        pi: numpy.ndarray of shape (k,) containing the cluster priors
        m: numpy.ndarray of shape (k, d) containing the centroid means
        S:numpy.ndarray of shape (k, d, d) containing the covariance matrices
        clss: numpy.ndarray of shape (n,) containing the cluster
            indices for each data point
        bic: numpy.ndarray of shape (kmax - kmin + 1) containing the BIC value
            for each cluster size tested
    """
    gmm = sklearn.mixture.GaussianMixture(n_components=k,).fit(X)

    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return (pi, m, S, clss, bic)
