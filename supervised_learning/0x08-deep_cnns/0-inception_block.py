#!/usr/bin/env python3
"""Module that  builds an inception block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """builds an inception block
    Args:
        A_prev: is the output from the previous layer
        filters: is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,

        F1 number of filters in the 1x1 convolution
        F3R number of filters in the 1x1 convolution before the 3x3 convolution
        F3  number of filters in the 3x3 convolution
        F5R number of filters in the 1x1 convolution before the 5x5 convolution
        F5  number of filters in the 5x5 convolution
        FPP number of filters in the 1x1 convolution after the max pooling
    Return: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal(seed=None)
    conv_1x1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)

    conv_1x1x3 = K.layers.Conv2D(
        filters=F3R,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)
    conv_3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(conv_1x1x3)

    conv_1x1x5 = K.layers.Conv2D(
        filters=F5R,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)

    conv_5x5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(conv_1x1x5)

    max_pool = K.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)

    conv_max_1x1 = K.layers.Conv2D(
        filters=FPP,
        kernel_size=1,
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(max_pool)

    module = K.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, conv_max_1x1])
    return module
