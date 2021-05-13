#!/usr/bin/env python3
""" Method that creates layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """ method to create layer """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        name='layer')
    y = layer(prev)
    return y
