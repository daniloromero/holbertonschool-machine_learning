#!/usr/bin/env python3
"""Module that sets up Adam optimization for a keras model """
from tensorflow import keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Adam optimization for a keras model with categorical crossentropy loss
     and accuracy metrics
    Args:
        network is the model to optimize
        alpha is the learning rate
        beta1 is the first Adam optimization parameter
        beta2 is the second Adam optimization parameter
    Returns: None
    """
    opt = K.optimizers.Adam(lr=alpha,  beta_1=beta1, beta_2=beta2)
    network.compile(
        optimizer=opt,

        metrics=['accuracy']
    )
