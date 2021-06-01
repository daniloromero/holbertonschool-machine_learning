#!/usr/bin/env python3
"""Module that converts a label vector into a one-hot matrix"""
from tensorflow import keras as K


def one_hot(labels, classes=None):
    """ converts a label vector into a one-hot matrix
    Args:
        labels: array tha contains the
        The last dimension of the one-hot matrix must be the number of classes
    Returns: the one-hot matrix
    """
    return K.utils.to_categorical(labels, classes)
