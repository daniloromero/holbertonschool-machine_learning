#!/usr/bin/env python3
"""Module that performs pooling on images:"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ performs max or avg pooling on images:
    Args:
        images: numpy.ndarray, shape (m, h, w) with multiple grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        kernel_shape: numpy.ndarray, shape (kh, kw) with kernel for convolution
            kh is the height of the kernel
            kw is the width of the kernel
        stride: is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
        mode indicates the type of pooling
            max indicates max pooling
            avg indicates average pooling
    Returns: a numpy.ndarray containing the pooled images
    """
    m, input_w, input_h, c = images.shape
    kernel_w, kernel_h = kernel_shape
    sh, sw = stride

    h_pad = int(((input_w - kernel_w) / sh) + 1)
    w_pad = int(((input_h - kernel_h) / sw) + 1)

    output = np.zeros((m, h_pad, w_pad, c))
    img_size = np.arange(m)
    for i in range(h_pad):
        for j in range(w_pad):
            s_i = i * sh
            s_j = j * sw
            window = images[img_size, s_i:kernel_w+s_i, s_j:kernel_h+s_j]
            if mode == 'max':
                output[img_size, i, j] = window.max(axis=(1, 2))
            if mode == 'avg':
                output[img_size, i, j] = window.mean(axis=(1, 2))
    return output
