#!/usr/bin/env python3
"""Module that that builds the inception network"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Build inception network"
    Args:
         the input data will have shape (224, 224, 3)
    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)
    layer_1 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    max_pool1 = K.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(layer_1)

    layer_2 = K.layers.Conv2D(
        filters=64,
        kernel_size=1,
        strides=1,
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(max_pool1)

    layer_3 = K.layers.Conv2D(
        filters=192,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(layer_2)

    max_pool2 = K.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(layer_3)

    layer_4 = inception_block(max_pool2, [64, 96, 128, 16, 32, 32])

    layer_5 = inception_block(layer_4, [128, 128, 192, 32, 96, 64])

    max_pool3 = K.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(layer_5)

    layer_6 = inception_block(max_pool3, [192, 96, 208, 16, 48, 64])

    layer_7 = inception_block(layer_6, [160, 112, 224, 24, 64, 64])

    layer_8 = inception_block(layer_7, [128, 128, 256, 24, 64, 64])

    layer_9 = inception_block(layer_8, [112, 144, 288, 32, 64, 64])

    layer_10 = inception_block(layer_9, [256, 160, 320, 32, 128, 128])

    max_pool4 = K.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(layer_10)

    layer_11 = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])

    layer_12 = inception_block(layer_11, [384, 192, 384, 48, 128, 128])

    l_avg_pool = K.layers.AveragePooling2D(pool_size=[7, 7],
                                           strides=7,
                                           padding='same')(layer_12)

    dropout = K.layers.Dropout(0.4)(l_avg_pool)

    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=init)(dropout)
    model = K.models.Model(inputs=X, outputs=Y)
    return model
