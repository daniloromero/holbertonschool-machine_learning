#!/usr/bin/env python3
"""Module that performs convolution on grayscale images with custom paddind"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ performs a same convolution on grayscale images with custom paddind
    Args:
        images: numpy.ndarray, shape (m, h, w) with multiple grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel: numpy.ndarray, shape (kh, kw) with kernel for the convolution
            kh is the height of the kernel
            kw is the width of the kernel
        padding is a tuple of (ph, pw)
            ph is the padding for the height of the image
            pw is the padding for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    m, input_w, input_h = images.shape
    kernel_w, kernel_h = kernel.shape
    ph, pw = padding

    h_pad = input_w + 2 * ph - kernel_w + 1
    w_pad = input_h + 2 * pw - kernel_h + 1

    input_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    output = np.zeros((m, h_pad, w_pad))
    img_size = np.arange(m)
    for i in range(input_w):
        for j in range(input_h):
            window = input_padded[img_size, i:kernel_w+i, j:kernel_h+j]
            output[img_size, i, j] = np.sum(window * kernel, axis=(1, 2))
    return output
