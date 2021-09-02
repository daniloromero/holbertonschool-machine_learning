#!/usr/bin/env python3
"""Module the creates GRUCell that represents gated rrecurrent unit"""
import numpy as np


class GRUCell:
    """class that represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """class constructor
        Args:
            i: is the dimensionality of the data
            h: is the dimensionality of the hidden state
            o: is the dimensionality of the outputs
        Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
            that represent the weights and biases of the cell
            Wz and bz are for the update gate
            Wr and br are for the reset gate
            Wh and bh are for the intermediate hidden state
            Wy and by are for the output
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        ''' sigmoid function '''
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(z):
        ''' softmax activation function '''
        t = np.exp(z)
        a = np.exp(z) / np.sum(t, axis=1).reshape(-1, 1)
        return a

    def forward(self, h_prev, x_t):
        """peforms forward propagation for one time step
        Args:
            x_t: numpy.ndarray shape(m, i) that contains the input data for cell
                m is the batch size
            h_prev: numpy.ndarray shape(m, h) containing previous hidden state
        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        m, i = x_t.shape
        cat = np.concatenate((h_prev, x_t), axis=1)
        z = self.sigmoid(cat @ self.Wz + self.bz)
        r = self.sigmoid(cat @ self.Wr + self.br)

        cat_r = np.concatenate(((h_prev * r), x_t), axis=1)
        h = np.tanh(cat_r @ self.Wh + self.bh)
        h_next = (np.ones_like(z) - z) * h_prev + z * h
        y = self.softmax(h_next @ self.Wy + self.by)
        return h_next, y
