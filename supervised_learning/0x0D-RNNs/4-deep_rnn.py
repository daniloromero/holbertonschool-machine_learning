#!/usr/bin/env python3
"""Module that perfomrs forward propagation for a deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs foward propagation for a deep RNN
    Args:
        rnn_cells: a list of RNNCells instances of lenght l to use
        X: is the data given as a numpy.ndarray shape(l, m, h)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0: is the initial hidden state, given as numpy.ndarray shape(l, m, h)
            h is the dimensionality of the hidden state
    Returns: H, Y
        H is numpy.ndarray containing all the hidden states
        Y is numpy.ndarray containing all the outputs
    """
    t, m, i = X.shape
    l, m, h = h_0.shape
    h_n = np.zeros((t + 1, l, m, h))
    y_pred = []

    h_next = h_0
    # loop over all time-steps
    for t_s in range(t):
        # loop over each layer l
        for l in range(len(rnn_cells)):
            x = X[t_s] if l is 0 else h_n[t_s + 1, l - 1]
            h_n[t_s + 1, l], y = rnn_cells[l].forward(h_next[l], x)
        h_next = h_n[t_s + 1]
        y_pred.append(y)
    return h_n, np.array(y_pred)
