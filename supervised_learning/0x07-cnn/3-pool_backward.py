#!/usr/bin/env python3
"""Module that performs back propagation over a pooling layer of a NN"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """"back propagation over a pooling layer of a NN
    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing partial
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
        kernel_shape: is a numpy.ndarray of shape (kh, kw)
            containing the kernels for the convolution
            kh is the filter height
            kw is the filter width
        stride: tuple of (sh, sw) containing the strides for the convolution
            sh is the stride for the height
            sw is the stride for the width
        mode: string either max or avg to perform pooling
    Returns: the partial derivatives with respect to the previous layer dA_prev
    """
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    s_h = h * sh
                    s_w = w * sw
                    box = dA[i, h, w, c]
                    if mode == 'max':
                        tmp = A_prev[i, s_h:kh+s_h, s_w:kw+s_w, c]
                        mask = (tmp == np.max(tmp))
                        dA_prev[i, s_h:kh+s_h, s_w:kw+s_w, c] += box * mask
                    if mode == 'avg':
                        dA_prev[i, s_h:kh+s_h, s_w:kw+s_w, c] += box/kh/kw
    return dA_prev
