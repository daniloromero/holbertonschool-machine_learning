#!/usr/bin/env python3
""" class NeuralNetwork """
import numpy as np


class NeuralNetwork:
    """ class NeuralNetwork with one hidden layer"""

    def __init__(self, nx, nodes):
        """ Neural Network class constructor """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0

    @property
    def W1(self):
        """Weight1 getter """
        return self.__W1

    @property
    def b1(self):
        """Bias 2 getter """
        return self.__b2

    @property
    def A1(self):
        """Activation 1 getter"""
        return self.__A1

    @property
    def W2(self):
        """Weight2 getter """
        return self.__W2

    @property
    def b2(self):
        """Bias 1 getter """
        return self.__b1

    @property
    def A2(self):
        """Weight1 getter """
        return self.__A2
