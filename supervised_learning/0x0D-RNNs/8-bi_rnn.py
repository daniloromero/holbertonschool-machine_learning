#!/usr/bin/env python3
"""Module that performs forward propagation for a bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ performs forward prop for a bidirectional RNN
    Args:
        bi_cell instance of BidirectinalCell that used for forward propagation
        X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0 is the initial hidden state in forward direction, ndarray  (m, h)
            h is the dimensionality of the hidden state
        h_t is the initial hidden state in backward direction, ndarray (m, h)
        Returns: H, Y
            H is a ndarray containing all of the concatenated hidden states
            Y is a ndarray containing all of the outputs
    """
    t, m, i = X.shape
    m, h = h_0.shape
    H = np.zeros((t, m, h * 2))
    F = np.zeros((t, m, h))
    B = np.zeros((t, m, h))
    Y_pred = []
    h_prev = h_0
    h_next = h_t
    # loop over all time-steps
    for t_s in range(t):
        # peerform forward propagationa
        F[t_s] = bi_cell.forward(h_prev, X[t_s])
        # perform backward propagation
        B[t - t_s - 1] = bi_cell.backward(h_next, X[t - t_s - 1])
        # update hidden states
        h_next = B[t - t_s - 1]
        h_prev = F[t_s]
    H = np.concatenate((F, B), axis=2)
    Y_pred = bi_cell.output(H)
    return H, np.array(Y_pred)
