#!/usr/bin/env python3
"""Module that performs forward propagation over a pooling layer of NN"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs forward propagation over a pooling layer of NN
    Args:
        A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        kernel_shape is a tuple of (kh, kw) size of the kernel for the pooling
        kh is the kernel height
        kw is the kernel width
        stride is a tuple of (sh, sw) containing the strides for the pooling
            sh is the stride for the height
            sw is the stride for the width
        mode is a string containing either max or avg, indicating whether
            to perform maximum or average pooling
    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_pad = int(((h_prev - kh) / sh) + 1)
    w_pad = int(((w_prev - kw) / sw) + 1)

    output = np.zeros((m, h_pad, w_pad, c_prev))
    img_size = np.arange(m)
    for i in range(h_pad):
        for j in range(w_pad):
            s_i = i * sh
            s_j = j * sw
            window = A_prev[img_size, s_i:kh+s_i, s_j:kw+s_j]
            if mode == 'max':
                output[img_size, i, j] = window.max(axis=(1, 2))
            if mode == 'avg':
                output[img_size, i, j] = window.mean(axis=(1, 2))
    return output
