#!/usr/bin/env python3
"""Module that flips an image horizontally"""
import tensorflow as tf


def flip_image(image):
    """Flips image horizontally
    Args:
        image is a 3D tensor containing the image to flip

    Returns the flipped image
    """
    return tf.image.flip_left_right(image)
