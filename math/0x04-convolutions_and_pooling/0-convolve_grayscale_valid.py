#!/usr/bin/env python3
"""Module that performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ performs a valid convolution on grayscale images
    Args:
        images: numpy.ndarray, shape (m, h, w) with multiple grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel: numpy.ndarray, shape (kh, kw) with kernel for the convolution
            kh is the height of the kernel
            kw is the width of the kernel
    Returns: a numpy.ndarray containing the convolved images
    """
    m, input_w, input_h = images.shape
    kernel_w, kernel_h = kernel.shape
    output_height = input_h - kernel_h + 1
    output_width = input_w - kernel_w + 1
    output = np.zeros((m, output_width, output_height))
    img_size = np.arange(m)
    for i in range(output_width):
        for j in range(output_height):
            window = images[img_size, i:kernel_h+i, j:kernel_w+j]
            output[img_size, i, j] = np.sum(window * kernel, axis=(1, 2))
    return output
