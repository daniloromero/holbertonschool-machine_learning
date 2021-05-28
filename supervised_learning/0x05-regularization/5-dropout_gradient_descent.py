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
    cache = {'A0': Y}
    for l in range(L):
        A = cache.get('A' + str(l))
        Weight = weights.get('W' + str(l + 1))
        Bias = weights.get('b' + str(l + 1))
        Z = np.matmul(Weight, A.T) + Bias
        if l < L - 1:
            A_next = (2/(1 + np.exp(-2 * Z)))
            drop = np.random.rand(A_next.shape[0], A_next.shape[1]) < keep_prob
            A_next = np.multiply(A_next, drop)
            cache['A' + str(l + 1)] = A_next / keep_prob
            cache['D' + str(l + 1)] = drop * 1
        else:
            softmax = np.exp(Z)
            cache['A' + str(l + 1)] = softmax / np.sum(softmax)
        return cache