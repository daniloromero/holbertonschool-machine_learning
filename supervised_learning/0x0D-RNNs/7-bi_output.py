#!/usr/bin/env python3
"""Module that creates a BidirectionalCell of an RNN"""
import numpy as np


class BidirectionalCell:
    """represents a bidirectional cell of a RNN"""

    def __init__(self, i, h, o):
        """class constructor
            i is the dimensionality of the data
            h is the dimensionality of the hidden states
            o is the dimensionality of the outputs
        Creates the public instance attributes Whf, Whb, Wy, bhf, bhb, by
            that represent the weights and biases of the cell

            Whf and bhfare for the hidden states in the forward direction
            Whb and bhbare for the hidden states in the backward direction
            Wy and byare for the outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h + h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(z):
        ''' softmax activation function '''
        t = np.exp(z)
        a = np.exp(z) / np.sum(t, axis=1).reshape(-1, 1)
        return a

    def forward(self, h_prev, x_t):
        """method that calculates hidden state in forw>> direction for timestep
        Args
            x_t is a numpy.ndarray of shape (m, i) that contains
                the data input for the cell
                m is the batch size for the data
            h_prev is a numpy.ndarray of shape (m, h) containing
                the previous hidden state
        Returns: h_next, the next hidden state
        """
        cat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(cat, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """method that calculates hidden state in forw>> direction for timestep
        Args
            x_t is a numpy.ndarray of shape (m, i) that contains
                the data input for the cell
                m is the batch size for the data
            h_prev is a numpy.ndarray of shape (m, h) containing
                the previous hidden state
        Returns: h_next, the next hidden state
        """
        cat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(cat, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """mthod that calculates ouputs for the RNN
        Args:
            H is a numpy.ndarray of shape (t, m, 2 * h) that contains the
                concatenated hidden states from both directions,
                excluding their initialized states
                t is the number of time steps
                m is the batch size for the data
                h is the dimensionality of the hidden states
        Returns: Y, the outputs
        """
        t, m, s = H.shape
        Y_pred = []
        for t_s in range(t):
            y = self.softmax(np.dot(H[t_s], self.Wy) + self.by)
            Y_pred.append(y)
        return np.array(Y_pred)
