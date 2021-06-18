#!/usr/bin/env python3
"""Module that bulids a dense block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """build a dense block
    Args:
        X: is the output from the previous layer
        nb_filters: is an integer representing the number of filters in X
        growth_rate: is the growth rate for the dense block
        layers: is the number of layers in the dense block
    Returns: The concatenated output of each layer
    """

    init = K.initializers.he_normal(seed=None)

    for layer in range(layers):
        batch_norm1 = K.layers.BatchNormalization(axis=3)(X)
        activation_1 = K.layers.Activation('relu')(batch_norm1)

        conv_1 = K.layers.Conv2D(
            filters=128,
            kernel_size=1,
            padding='same',
            kernel_initializer=init
        )(activation_1)

        batch_norm2 = K.layers.BatchNormalization(axis=3)(conv_1)
        activation_2 = K.layers.Activation('relu')(batch_norm2)

        conv_2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding='same',
            kernel_initializer=init
        )(activation_2)

        X = K.layers.concatenate([X, conv_2])
        nb_filters += growth_rate
    return X, nb_filters
