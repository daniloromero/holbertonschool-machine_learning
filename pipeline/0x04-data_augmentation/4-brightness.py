#!/usr/bin/env python3
"""Module that randomly changes he brightmess of an image"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly changes the brightness of an image
    Args:
        image is a 3D tf.tensor containing the image to change
        max_delta is  the maximum amount the image should be brightened
    Return the altered image
    """
    return tf.image.adjust_brightness(image, max_delta)
