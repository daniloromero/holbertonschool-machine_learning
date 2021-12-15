#!/usr/bin/env python3
"""MOdule that changes he hue of an image"""
import tensorflow as tf


def change_hue(image, delta):
    """Perfomrs changes in hue of image
    Args:
        image is a 3D tf.tensor containing the image to change
        delta is the amount the hue should change
    Return the altered image
    """
    return tf.image.adjust_hue(image, delta)
