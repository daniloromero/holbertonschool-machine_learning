#!/usr/bin/env python3
""""Module that determines if a markov chain is absorbing"""
import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing
    Args:
    P: square 2D numpy.ndarray shape (n, n) representing the transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    Return: True if it is absorbing, or False on failure
    """
    p1, p2 = P.shape
    if p1 != p2 or type(P) != np.ndarray:
        return False
    if len(P.shape) != 2:
        return None
    if not np.any(np.diag(P) == 1):
        return False
    return True         
