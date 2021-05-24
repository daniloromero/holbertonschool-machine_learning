#!/usr/bin/env python3
"""Module that updates the learning rate using inverse time decay in numpy"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ updates the learning rate using inverse time decay in numpy
    Args:
        alpha: is the original learning rate
        decay_rate: weight used to set the rate at which alpha will decay
        global_step: number of passes of gradient descent that have elapsed
        decay_step: is the number of passes of gradient descent that occurs
            before alpha is decayed further
    Returns: the updated value for alpha
    """
    alpha /= (1 + decay_rate * np.floor(global_step / decay_step))
    return alpha
