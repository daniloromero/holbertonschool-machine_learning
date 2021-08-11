#!/usr/bin/env python3
"""Module the determines the probability of markov chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """ determines the probability of a markov chain being in a particular
        state after a specified number of iterations:
    Args:
    P: square 2D numpy.ndarray shape (n, n) representing the transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    s: numpy.ndarray shape (1, n) representing the probability
        of starting in each state
    t: number of iterations that the markov chain has been through
    Returns: numpy.ndarray shape (1, n) representing the probability
        of being in a specific state after t iterations, or None on failure
    """
    if type(P) != np.ndarray or type(s) != np.ndarray:
        return None
    p1, p2 = P.shape
    if p1 != p2:
        return None
    if len(P.shape) != 2:
        return None
    s1, s2 = s.shape
    if s1 != 1 or s2 != p1:
        return None
    for i in range(t):
        s = s @ P

    return s
