#!/usr/bin/env python3
"""Module that conducts forward propagation using Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """forward propagation using Dropout
    Args:
        X: a numpy.ndarray of shape (nx, m) with input data for the network
            nx: is the number of input features
            m: is the number of data points

        weights: is a dictionary of the weights and biases of neural network
        L: the number of layers in the network
        keep_prob: is the probability that a node will be kep
    """
    m = X.shape[1]
    cache = {'A0': X}
    for l in range(L):
        A = cache.get('A' + str(l))
        Weight = weights.get('W' + str(l + 1))
        Bias = weights.get('b' + str(l + 1))
        Z = np.matmul(Weight, A) + Bias
        drop = np.random.binomial(1, keep_prob, size=Z.shape)
        # drop = np.where(drop < keep_prob, 1, 0)
        if l < L - 1:
            A_next = np.tanh(Z)
            A_next = np.multiply(A_next, drop)
            cache['A' + str(l + 1)] = A_next / keep_prob
            cache['D' + str(l + 1)] = drop
        else:
            softmax = np.exp(Z)
            cache['A' + str(l + 1)] = softmax / np.sum(softmax,
                                                       axis=0, keepdims=True)
    return cache
