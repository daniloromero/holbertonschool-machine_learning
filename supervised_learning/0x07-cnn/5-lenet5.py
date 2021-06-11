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
    i_shape = X.shape
    print(i_shape)
    init = K.initializers.he_normal(seed=None)
    model = K.Sequential()
    model.add(K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        input_shape=i_shape[1:],
        kernel_initializer=init,
    ))

    model.add(K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    model.add(K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init,
    ))

    model.add(K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    model.add(K.layers.Flatten())

    model.add(K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init,
    ))

    model.add(K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init,
    ))

    model.add(K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init,
    ))

    opt = K.optimizers.Adam()

    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
