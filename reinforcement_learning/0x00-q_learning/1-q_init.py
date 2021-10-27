#!/usr/bin/env python3
"""Module that initialize the Q-table"""
import gym
import numpy as np


def q_init(env):
    """Initializes theQ-table
    Args:
        env is the FrozenLakeEnv instance
    Returns: the Q-table as a numpy.ndarray of zeros
    """
    action_space = env.action_space.n
    state_space = env.observation_space.n
    q_table = np.zeros((state_space, action_space))
    return q_table
