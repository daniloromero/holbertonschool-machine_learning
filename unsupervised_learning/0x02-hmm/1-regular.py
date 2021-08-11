#!/usr/bin/env python3
"""Module determines the steady state probabilities of regular markov chain"""
import numpy as np


def augment_P(P):
    dimension = P.shape[0]
    M = np.vstack((P.transpose()[:-1], np.ones(dimension)))
    b = np.vstack((np.zeros((dimension - 1, 1)), [1]))
    return M, b


def regular(P):
    """determines the steady state probabilities of a regular markov chain
    Args:
    P: square 2D numpy.ndarray shape (n, n) representing the transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady state
        probabilities, or None on failure
    """
    if type(P) != np.ndarray:
        return None
    p1, p2 = P.shape
    if p1 != p2:
        return None
    prob = np.ones((1, p1))
    if not (np.isclose((np.sum(P, axis=1)), prob)).all():
        return (None)
    if len(P.shape) != 2:
        return (None)
    if not (P > 0).all():
        return None
    dim = P.shape[0]
    q = (P - np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q, ones]
    QTQ = np.dot(q, q.T)
    bQT = np.ones(dim)
    return np.expand_dims(np.linalg.solve(QTQ, bQT), axis=0)
