#!/usr/bin/env python3
"""Module that Create the class LSTMCell that represents an LSTM"""
import numpy as np


class LSTMCell:
    """ represents a LTSM unit"""

    def __init__(self, i, h, o):
        """class constuctor
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
(i, o) is the dimensionality of the outputs
            Creates the public instance attributes
                    Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by
                Wf and bf are for the forget gate
                Wu and bu are for the update gate
                Wc and bc are for the intermediate cell state
                Wo and bo are for the output gate
                Wy and by are for the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """perfomrs foward propagation
        Args:
            x_t is a numpy.ndarray of shape (m, i) that contains
                the data input for the cell
                m is the batche size for the data
            h_prev is a numpy.ndarray of shape (m, h) containing
                the previous hidden state
            c_prev is a numpy.ndarray of shape (m, h) containing
                the previous cell state
        Returns: h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell
        """
        # concatenae previous hidden state + input
        cat = np.concatenate((h_prev, x_t), axis=1)
        # calculate forget activation vector
        ft = self.sigmoid(cat @ self.Wf + self.bf)
        # calculate update activation vector
        ut = self.sigmoid(cat @ self.Wu + self.bu)
        # cell input activation vector
        c = np.tanh(cat @ self.Wc + self.bc)
        # cell state vector
        c_next = ft * c_prev + ut * c
        ot = self.sigmoid(cat @ self.Wo + self.bo)
        # next hidden state
        h_next = ot * np.tanh(c_next)
        # cell activate output
        y = self.softmax(h_next @ self.Wy + self.by)
        return h_next, c_next, y
