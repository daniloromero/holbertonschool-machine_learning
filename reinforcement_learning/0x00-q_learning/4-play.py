#!/usr/bin/env python3
""" Play """
import numpy as np


def play(env, Q, max_steps=100):
    """ Play """
    total_rewards = 0
    state = env.reset()
    env.render()
    for step in range(max_steps):

        action = np.argmax(Q[state, :])

        next_state, reward, done, info = env.step(action)

        env.render()

        state = next_state
        total_rewards += reward
        if done:
            break
    env.close()
    return total_rewards
