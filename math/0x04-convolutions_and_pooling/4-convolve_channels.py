#!/usr/bin/env python3
"""Module that performs strided convolution on images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """performs a strided convolution on images with channles
    Args:
        images: numpy.ndarray, shape (m, h, w) with multiple grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        kernel: numpy.ndarray, shape (kh, kw) with kernel for the convolution
            kh is the height of the kernel
            kw is the width of the kernel
        padding: is a tuple of (ph, pw)
            ph is the padding for the height of the image
            pw is the padding for the width of the image
        stride: is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    m, input_w, input_h, c = images.shape
    kernel_w, kernel_h, c = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph = int((((input_w - 1) * sh + kernel_w - input_w) / 2) + 1)
        pw = int((((input_h - 1) * sw + kernel_h - input_h) / 2) + 1)
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    h_pad = int(((input_w + 2 * ph - kernel_w) / sh) + 1)
    w_pad = int(((input_h + 2 * pw - kernel_h) / sw) + 1)

    input_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    output = np.zeros((m, h_pad, w_pad))
    img_size = np.arange(m)
    for i in range(h_pad):
        for j in range(w_pad):
            s_i = i * sh
            s_j = j * sw
            window = input_padded[img_size, s_i:kernel_w+s_i, s_j:kernel_h+s_j]
            output[img_size, i, j] = np.sum(window * kernel, axis=(1, 2, 3))
    return output
