#!/usr/bin/env python3
"""Module that saves and loads a model"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """saves a model’s weights
    Args:
        network: the model whose weights should be saved
        filename: path of the file that the weights should be saved to
        save_format: the format in which the weights should be saved
    Returns: None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """loads a model’s weights
    Args:
        network: is the model to which the weights should be loaded
        filename:path of the file that the weights should be loaded from
    Returns: None
    """
    network.load_weights(filename)
