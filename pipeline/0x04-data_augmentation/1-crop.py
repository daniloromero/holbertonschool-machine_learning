#!/usr/bin/env python3
"""Module that performs a rondom crop of an image"""
import tensorflow as tf


def crop_image(image, size):
    """"Performs a random crop of an image
    Args:
        imge is a 3D tf.tensor containing the image to crop
        size is a tuple containing the size of the crop
    Return the cropped image
    """
    return tf.image.random_crop(image, size)
