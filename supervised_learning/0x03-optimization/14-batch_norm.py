#!/usr/bin/env python3
"""normalizes an unactivated output of a NN layer using batch normalization"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """that creates a batch normalization layer for a NN in tensorflow
    Args:
        prev: is the activated output of the previous layer
        n: is the number of nodes in the layer to be created
        activation: activation function to be used on the output of the layer
        layer uses 2 trainable parameters, gamma,  beta, as vectors of 1 and 0
        epsilon:  1e-8
    Returns: a tensor of the activated output for the layer
    """
    variance_epsilon = 1e-8
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    base_layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        name='base_layer'
    )
    y = base_layer(prev)
    mean, variance = tf.nn.moments(y, axes=[0])
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    batch_normalization = tf.nn.batch_normalization(
        y,
        mean,
        variance,
        gamma,
        beta,
        variance_epsilon
    )
    return activation(batch_normalization)
