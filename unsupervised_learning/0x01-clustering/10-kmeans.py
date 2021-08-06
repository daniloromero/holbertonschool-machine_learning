#!/usr/bin/env python3
"""Module that performs K-means on a data set"""
import sklearn.cluster


def kmeans(X, k):
    """Performs K-means on a data set
    Args:
        X is a numpy.ndarray of shape (n, d) containing the dataset
        k is the number of clusters
    Returns: C, clss
        C is a numpy.ndarray of shape (k, d) containing the centroid means
            for each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of the
            cluster in C that each data point belongs to
    """
    kmean = sklearn.cluster.KMeans(n_clusters=k)
    kmean.fit(X)
    clss = kmean.labels_
    C = kmean.cluster_centers_
    return (C, clss)
