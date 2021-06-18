#!/usr/bin/env python3
""" Module that builds a modified version of the LeNet-5 architecture"""
import tensorflow.keras as K


def lenet5(X):
    """the LeNet-5 architecture
    Args:
        X:is a K.Input of shape (m, 28, 28, 1) with input images for the NN
            m is the number of images
    Return: a K.Model compiled to use Adam optimization and accuracy metrics
    """
    init = K.initializers.he_normal(seed=None)
    conv_l1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        input_shape=(28, 28, 1),
        kernel_initializer=init,
    )(X)

    max_pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv_l1)

    conv_l2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init,
    )(max_pool1)

    max_pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv_l2)

    flatten = K.layers.Flatten()(max_pool2)

    fc_l1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init,
    )(flatten)

    fc_l2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init,
    )(fc_l1)

    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init,
    )(fc_l2)

    model = K.Model(inputs=X, outputs=output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
