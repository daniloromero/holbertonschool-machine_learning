#!/usr/bin/env/ python3
""" Module defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """class for deep neural network performing binary classification """

    @staticmethod
    def Weights_init(nx, layers):
        """ Het-at-al Initialization of Weigths"""
        weights_I = {}
        for l in range(1, len(layers)):
            if type(layers[l]) is not int or layers[l] < 1:
                raise TypeError('layers must be a list of positive integers')
            prev_layer = layers[l - 1] if l > 0 else nx
            weights_I['W' + str(l)] = np.random.randn(
                layers[l], prev_layer) * np.sqrt(2/prev_layer)
            weights_I['b' + str(l)] = np.zeros((layers[l], 1))
        return weights_I

    def __init__(self, nx, layers):
        """ Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        self.L = len(layers)
        self.cache = {}
        self.weights = self.Weights_init(nx, layers)
