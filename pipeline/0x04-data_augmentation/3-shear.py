#!/usr/bin/env python3
"""Module that randomly shears an image"""
import tensorflow as tf


def shear_image(image, intensity):
    """Performs a random shear toa n image
    Args:
        image is a 3D tf.tensor containing the image to shear
        intensity is the intensity with which the image should be sheared
    Return the sheared image
    """
    return tf.keras.preprocessing.image.random_shear(image, intensity)
