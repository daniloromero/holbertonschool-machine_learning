#!/usr/bin/env python3
"""Module that updates weights with gradient descent with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates weights with gradient descent with L2 regularization
    Args:
        Y: one-hot numpy.ndarray shape (classes, m)contains correct labels
            classes is the number of classes
            m is the number of data points

        weights: is a dictionary of the weights and biases
        cache: is a dictionary of the outputs of each layer
        alpha: is the learning rate
        lambtha: is the L2 regularization parameter
        L: is the number of layers of the network
    """
    m = Y.shape[1]
    dz_prev = []
    copy_weights = weights.copy()
    for n in range(L, 0, -1):
        A = cache.get('A' + str(n))
        A_prev = cache.get('A' + str(n - 1))
        wx = copy_weights.get('W' + str(n + 1))
        bx = copy_weights.get('b' + str(n))
        if n == L:
            dz = A - Y
        else:
            dz = np.matmul(wx.T, dz_prev) * (1 - (A * A))
        dw = np.matmul(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        dz_prev = dz
        w = copy_weights.get('W' + str(n))
        weights['W' + str(n)] = w * (1 - (alpha * lambtha) / m) - (alpha * dw)
        weights['b' + str(n)] = bx - (db * alpha)
