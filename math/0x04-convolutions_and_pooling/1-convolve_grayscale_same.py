#!/usr/bin/env python3
"""Module that performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ performs a same convolution on grayscale images
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

    if kernel_w % 2 == 0:
        pw = int(kernel_w / 2)
    else:
        pw = int((kernel_w - 1) / 2)

    if kernel_h % 2 == 0:
        ph = int(kernel_h / 2)
    else:
        ph = int((kernel_h - 1) / 2)

    input_padded = np.pad(images, ((0, 0), (pw, pw), (ph, ph)), 'constant')

    output = np.zeros((m, input_w, input_h))
    img_size = np.arange(m)
    for i in range(input_w):
        for j in range(input_h):
            window = input_padded[img_size, i:kernel_w+i, j:kernel_h+j]
            output[img_size, i, j] = np.sum(window * kernel, axis=(1, 2))
    return output
