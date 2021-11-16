#!/usr/bin/env python3
"""Module that perfomrs Sarsa(λ)"""
import numpy as np


def epsilon_greedy(env, Q, state, epsilon):
    """ performs epsilon greedy policy """
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000,
                  max_steps=100, alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """performs the TD(λ) algorithm
        env is the openAI environment instance
        Q is a numpy.ndarray of shape (s,) containing the Q table
        lambtha is the elgibility trace factor
        episodes is the total number of episodes to train over
        max_steps is the maximum number of steps per episode
        alpha is the learning rate
        gamma is the discount rate
        epsilon is the initial threshold for epsilon greedy
        min_epsilon is the min value epsilon should decay to
        epsilon_decay is decay rate for updating epsilon between episodes
        Returns: Q, the updated value estimate
    """
    initial_epsilon = epsilon
    states = Q.shape[0]
    elig_traces = np.zeros(Q.shape)

    for e in range(episodes):
        s = env.reset()
        action = epsilon_greedy(env, Q, s, epsilon)
        for steo in range(max_steps):
            next_state, reward, done, info = env.step(action)
            next_action = epsilon_greedy(env, Q, s, epsilon)

            elig_traces *= gamma * epsilon
            elig_traces[s, action] += (1.0)

            delta = reward + gamma * Q[next_state, next_action] - Q[s, action]
            Q += alpha * delta * elig_traces

            if done:
                break
            else:
                s = next_state
                action = next_action
        if epsilon < min_epsilon:
            epsilon = min_epsilon
        else:
            epsilon *= initial_epsilon * np.exp((-epsilon_decay * e))
    return Q
