#!/usr/bin/env python3
"""Module that rotates an image by 90 degrees counter clockwise"""
import tensorflow as tf


def rotate_image(image):
    """Rotates an image 90 degrees counter clockwise
    Args:
        image is a 3D tf.tensor cointinin the image to rotate
    Return the rotated image
    """
    return tf.image.rot90(image)
