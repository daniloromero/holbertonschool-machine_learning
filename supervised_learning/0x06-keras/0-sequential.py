#!/usr/bin/env python3
"""Module that builds a neural network with the Keras library"""
import tensorflow.keras as K


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
    L2_reg = K.regularizers.l2(lambtha)
    model = K.Sequential()
    for i, (layer, activation) in enumerate(zip(layers, activations)):
        model.add(K.layers.Dense(
            layer, input_shape=(nx,),
            activation=activation,
            kernel_regularizer=L2_reg))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(rate=1 - keep_prob))
    return model
