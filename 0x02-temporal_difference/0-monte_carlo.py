#!/usr/bin/env python3
"""Module that perfomrs the Monte Carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """performs Monte Carlo algorith
    Args:
        env is the openAI environment instance
        V is a numpy.ndarray of shape (s,) containing the value estimate
        policy is a function that takes in a state and
            returns the next action to take
        episodes is the total number of episodes to train over
        max_steps is the maximum number of steps per episode
        alpha is the learning rate
        gamma is the discount rate
    Returns: V, the updated value estimate
    """
    for e in range(episodes):
        s = env.reset()
        st_in_ep = []
        for step in range(max_steps):
            action = policy(s)
            next_state, reward, done, info = env.step(action)
            st_in_ep.append((s, action, reward))
            if done:
                break

            s = next_state
        st_in_ep = np.array(st_in_ep, dtype=int)
        G = 0
        for j, step in enumerate(st_in_ep[::-1]):
            s, action, reward = step
            G = gamma * G + reward
            if s not in st_in_ep[:e, 0]:
                V[s] = V[s] + alpha * (G - V[s])
    return V
