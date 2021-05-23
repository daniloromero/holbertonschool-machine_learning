#!/usr/bin/env python3
"""Module creates the training operation for a neural network in tensorflow
 using the gradient descent with momentum optimization algorithm
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """creates the training operation for a neural network
    Args:
        loss is the loss of the network
        alpha is the learning rate
        beta1 is the momentum weight
    Returns: the momentum optimization operation
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    grads_vars = optimizer.compute_gradients(loss)
    operation = optimizer.apply_gradients(grads_vars)
    return operation
