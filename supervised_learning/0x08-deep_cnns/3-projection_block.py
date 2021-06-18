#!/usr/bin/env python3
"""Module that builds a projection block"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """builds a projection block
    Args:
        A_prev: is the output from the previous layer
        filters: is a tuple or list containing F11, F3, F12
            F11 is the number of filters in the first 1x1 convolution
            F3 is the number of filters in the 3x3 convolution
            F12 is the number of filters in the second 1x1 convolution
            s: is the stride of the first convolution
    Returns: the activated output of the projection block
        """
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=None)
    conv_1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        strides=s,
        padding='same',
        kernel_initializer=init
    )(A_prev)

    batch_norm1 = K.layers.BatchNormalization(axis=3)(conv_1)

    activation_1 = K.layers.Activation('relu')(batch_norm1)

    conv_2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding='same',
        kernel_initializer=init
    )(activation_1)

    batch_norm2 = K.layers.BatchNormalization(axis=3)(conv_2)

    activation_2 = K.layers.Activation('relu')(batch_norm2)

    conv_3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        padding='same',
        kernel_initializer=init
    )(activation_2)

    conv_4 = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        padding='same',
        strides=s,
        kernel_initializer=init
    )(A_prev)

    batch_norm3 = K.layers.BatchNormalization(axis=3)(conv_3)
    batch_norm4 = K.layers.BatchNormalization(axis=3)(conv_4)

    add = K.layers.Add()([batch_norm3, batch_norm4])

    module = K.layers.Activation('relu')(add)
    return module
