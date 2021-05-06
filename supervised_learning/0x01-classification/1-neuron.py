#!/usr/bin/env python3
""" class Neuron, defines a single neuron performing binary classification"""
import numpy as np


class Neuron:
    """ class Neuron, defines single neuron performing binary classification"""

    def __init__(self, nx):
        """ Neuron class constructor """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Weight getter """
        return self.__W

    @property
    def b(self):
        """ Bias getter"""
        return self.__b

    @property
    def A(self):
        """ Activation value """
        return self.__A
