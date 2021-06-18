#!/usr/bin/env python3
"""Module that  performs forward propagation over a convolutional layer NN"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ performs forward propagation over a convolutional layer of a NN
    Args:
        A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        W: is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
            containing the kernels for the convolution
            kh is the filter height
            kw is the filter width
            c_prev is the number of channels in the previous layer
            c_new is the number of channels in the output
        b: is a numpy.ndarray of shape (1, 1, 1, c_new)
            containing the biases applied to the convolution
        activation: is an activation function applied to the convolution
        padding: string, either same or valid, the type of padding used
        stride: tuple of (sh, sw) containing the strides for the convolution
            sh is the stride for the height
            sw is the stride for the width
    Returns: the output of the convolutional layer
    """
    m, h_prev, w_pev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_pev - 1) * sw + kw - w_pev) / 2)
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    h_pad = int(((h_prev + 2 * ph - kh) / sh) + 1)
    w_pad = int(((w_pev + 2 * pw - kw) / sw) + 1)

    input_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                          'constant')

    output = np.zeros((m, h_pad, w_pad, c_new))
    img_size = np.arange(m)
    for k in range(c_new):
        for i in range(h_pad):
            for j in range(w_pad):
                s_i = i * sh
                s_j = j * sw
                window = input_padded[img_size, s_i:kh+s_i,
                                      s_j:kw+s_j, :]
                kernel = W[:, :, :, k]
                output[img_size, i, j, k] = np.sum(window * kernel,
                                                   axis=(1, 2, 3,))
    Z = output + b
    return activation(Z)
