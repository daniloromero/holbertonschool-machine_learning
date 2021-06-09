#!/usr/bin/env python3
"""Module that performs back propagation over a convolutional layer of NN"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """performs back propagation over a convolutional layer of NN
    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing partial
            derivatives to the unactivated output of the convolutional layer
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c_new is the number of channels in the output
        A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        W: is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
            containing the kernels for the convolution
            kh is the filter height
            kw is the filter width
        b: is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        padding: string that is same or valid, indicating the type of padding
        stride: tuple of (sh, sw) containing the strides for the convolution
            sh is the stride for the height
            sw is the stride for the width
    Returns: the partial derivatives with respect to the previous layer
        (dA_prev), the kernels (dW), and the biases (db)
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = int((((h_prev - 1) * sh + kh - h_prev) / 2) + 1)
        pw = int((((w_prev - 1) * sw + kw - w_prev) / 2) + 1)
    if padding == 'valid':
        ph = 0
        pw = 0

    h_pad = int(((h_prev + 2 * ph - kh) / sh) + 1)
    w_pad = int(((w_prev + 2 * pw - kw) / sw) + 1)

    input_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                          'constant')
    _, h_new, w_new, c_new = dZ.shape
    dA_prev = np.zeros(input_padded.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    size = np.arange(m)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    s_h = h * sh
                    s_w = w * sw
                    box = dZ[i, h, w, c]
                    dA_prev[i, s_h:kh+s_h, s_w:kw+s_w] += box * W[:, :, :, c]
                    dW[:, :, :, c] += input_padded[i, s_h:kh+s_h,
                                                   s_w:kw+s_w, :] * box
    if padding == 'same':
        dA_prev = dA_prev[:, ph: -ph, pw:-pw, :]
    return dA_prev, dW
