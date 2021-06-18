#!/usr/bin/env python3
"""Module that creates a learning rate decay operation in tensorflow"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """learning rate decay operation in tensorflow using inverse time decay
    Args:
        alpha: is the original learning rate
        decay_rate: weight used to set the rate at which alpha will decay
        global_step: number of passes of gradient descent that have elapsed
        decay_step is the number of passes of gradient descent that occurs
            before alpha is decayed further
    Returns: the learning rate decay operation
    """
    return tf.train.inverse_time_decay(alpha,
                                       global_step,
                                       decay_step,
                                       decay_rate,
                                       staircase=True)
