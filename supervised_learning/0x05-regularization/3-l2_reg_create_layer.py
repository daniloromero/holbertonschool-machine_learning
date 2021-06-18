#!/usr/bin/env python3
"""Module that creates a tensorflow layer that includes L2 regularization:"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """t creates a tensorflow layer that includes L2 regularization
    Args:
        prev: is a tensor containing the output of the previous layer
        n: is the number of nodes the new layer should contain
        activation: is the activation function that should be used on the layer
        lambtha: is the L2 regularization parameter
    Returns: the output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    L2_reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=L2_reg
    )
    return layer(prev)
