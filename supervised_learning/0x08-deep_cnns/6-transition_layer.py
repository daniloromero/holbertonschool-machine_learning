#!/usr/bin/env python3
"""Module  that builds a transition layer"""
import numpy as np
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """builds a transition layer
    Args:
        X: is the output from the previous layer
        nb_filters: is an integer representing the number of filters in X
        compression: is the compression factor for the transition layer
    Return: The output of the transition layer and the number of filters
    """
    init = K.initializers.he_normal(seed=None)
    nb_filters = int(nb_filters * compression)
    batch_norm = K.layers.BatchNormalization(axis=3)(X)

    activation = K.layers.Activation('relu')(batch_norm)

    conv1 = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=1,
        padding='same',
        strides=1,
        kernel_initializer=init
    )(activation)
    l_avg_pool = K.layers.AveragePooling2D(pool_size=[2, 2],
                                           strides=2,
                                           padding='same')(conv1)

    return l_avg_pool, nb_filters
