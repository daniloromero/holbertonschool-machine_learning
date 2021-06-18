#!/usr/bin/env python3
"""Module that saves and loads a model"""
import tensorflow.keras as K


def save_model(network, filename):
    """saves an entire model
    Args:
        network: is the model to save
        filename: is the path of the file that the model should be saved to
    Returns: None
    """
    network.save(filename)


def load_model(filename):
    """loads an entire model
    Args:
        filename is the path of the file that the model should be loaded from
    Returns: the loaded model
    """
    return K.models.load_model(filename)
