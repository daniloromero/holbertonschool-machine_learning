#!/usr/bin/env python3
"""Module that uses epsilon-greedy to determine the next action"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Determine next action using epsilon-greedy
    Args:
        Q is a numpay.ndarray containing the q-table
        state is the current state
        epsilon is he epsilon to use for the calculation
    You should sample p with numpy.random.uniformn to determine if your
        algorithm should explore or exploit
    If exploring, you should pick the next action with
        numpy.random.randint from all possible actions
    Returns: the next action index
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.randint(Q.shape[1])  # Explore action space
    else:
        action = np.argmax(Q[state])  # Exploit learned values
    return action
