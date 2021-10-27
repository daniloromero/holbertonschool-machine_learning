#!/usr/bin/env python3
"""Module that performs Q-learning"""
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """performs Q-learning
    Args:
        env is the FrozenLakeEnv instance
        Q is a numpy.ndarray containing the Q-table
        episodes is the total number of episodes to train over
        max_steps is the maximum number of steps per episode
        alpha is the learning rate
        gamma is the discount rate
        epsilon is the initial threshold for epsilon greedy
        min_epsilon is the minimum value that epsilon should decay to
        epsilon_decay is the decay rate for updating epsilon between episodes
        When the agent falls in a hole, the reward should be updated to be -1
    Returns: Q, total_rewards
        Q is the updated Q-table
        total_rewards is a list containing the rewards per episode
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        rewards = 0
        for step in range(max_steps):

            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(Q.shape[1])  # Explore action space
            else:
                action = np.argmax(Q[state, :])  # Exploit learned values

            next_state, reward, done, info = env.step(action)
            old_value = Q[state, action]
            next_max = np.max(Q[next_state, :])

            new_value = (1 - alpha) * old_value + alpha *\
                        (reward + gamma * next_max)

            Q[state, action] = new_value
            rewards += reward
            state = next_state

            if done:
                break
        epsilon = min_epsilon + (epsilon - min_epsilon) *\
            np.exp(-epsilon_decay * episode)
        total_rewards.append(rewards)
    env.close()
    return Q, total_rewards
