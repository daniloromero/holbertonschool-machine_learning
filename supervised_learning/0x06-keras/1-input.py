#!/usr/bin/env python3
"""Module that builds a neural network with the Keras library"""
from tensorflow import keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library
    Args:
        nx: is the number of input features to the network
        layers: list containing the number of nodes in each layer
        activations: list containing activation functions used for each layer
        lambtha: is the L2 regularization parameter
        keep_prob: is the probability that a node will be kept for dropout
    Returns: the keras model
    """
    inputs = K.Input(shape=(1, nx))
    L2_reg = K.regularizers.l2(lambtha)
    for i, (layer, activation) in enumerate(zip(layers, activations)):
        x = K.layers.Dense(
            units=layer,
            activation=activation,
            kernel_regularizer=L2_reg
        )(x if i else inputs)
    model = K.Model(inputs=inputs, outputs=x)
    return model
