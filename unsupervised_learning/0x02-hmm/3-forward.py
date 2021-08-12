#!/usr/bin/env python3
"""Module that performs forward algorithm for a hidden markov model"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Performs forward algorithm for a hidden markov model
    Args:
        Observation: numpy.ndarray shape(T,) with index of observation
            T: number of observations
        Emission: numpy.ndarray shape(N, M) with emission probability of
            specific observation given a hidden state
            Emision[i, j] probability of observing j given a hidden state i
            N: number of hidden states
            M: number of all possible observations
        Transition: 2D numpy.ndarray shape(N, N) with transition probabilities
            Transition[i, j] probability of transitioning from state i to j
        Initial: numpy.ndarray shape(N, 1) with probability of starting in
            a particular hidden state
    Returns: P, F, or None, None on failure
        P: likelihood of the observations given the model
        F: numpy.ndarray shape(N, T) with the forward path probabilities
            F[i, j] probability of being in hidden state i at time j given
                previous observation
    """
    T = Observation.shape
    N, M = Emission.shape
    Nt1, Nt2 = Transition.shape
    Ni1, i2 = Initial.shape

    if type(Observation) != np.ndarray or len(T) > 1:
        return None, None
    if type(Emission) != np.ndarray:
        return None, None
    if type(Transition) != np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) != np.ndarray or len(Initial.shape) != 2:
        return None, None
    if N != Nt1 or Nt1 != Nt2 or Ni1 != N:
        return None, None
