#!/usr/bin/env python3
"""Module that updates weights of NN with Dropout reg with gradient descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates weights of NN with Dropout reg with gradient descent
    Args:
        Y is a one-hot numpy.ndarray of shape (classes, m) with correct labels
            classes is the number of classes
            m is the number of data points
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs and dropout masks of each layer of NN
        alpha: is the learning rate
        keep_prob: is the probability that a node will be kept
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
            dz *= cache['D{}'.format(n)]
            dz /= keep_prob
        dw = np.matmul(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        dz_prev = dz
        w = copy_weights.get('W' + str(n))
        weights['W' + str(n)] = w - (dw * alpha)
        weights['b' + str(n)] = bx - (db * alpha)
        dz_prev = dz
