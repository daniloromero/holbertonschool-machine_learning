#!/usr/bin/env python3
"""MOdule that performs forward ropagation for a simple RNN"""
import numpy as np


def rnn(rnn_cell, x, h_0):
    """performs forward propagation for a simple RNN
    Args:
        rnn_cell: an instance of RNNCell that will be used for the forward prop
        X: is the data to be used, given as a numpy.ndarray of shape (t, m, i)
            t is the maximum number ot times steps
            m is the batch size
            i is the dimensionality of the data
        h_0: is the initial hidden state, given as numpy.ndarray of shape(m, h)
            h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all the hidden states
        Y is a numpy.ndarray containing all the outputs
    """
    t, m, i = x.shape
    y_s, h = h_0.shape
    h_n = np.zeros((t + 1, y_s, h))
    y_pred = []

    h_next = h_0
    # loop over all time-steps
    for t_s in range(t):
        # Update hidden state, compute the prediction
        h_n[t_s + 1], y = rnn_cell.forward(h_next, x[t_s])
        h_next = h_n[t_s + 1]
        y_pred.append(y)
        print(y.shape)
    print(len(y_pred[0][0]))
    return h_n, np.array(y_pred)
