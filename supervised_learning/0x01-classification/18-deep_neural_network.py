#!/usr/bin/env python3
""" Module defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """class for deep neural network performing binary classification """

    @staticmethod
    def Weights_init(nx, layers):
        """ Het-at-al Initialization of Weigths"""
        weights_I = {}
        for l in range(len(layers)):
            if type(layers[l]) is not int or layers[l] < 1:
                raise TypeError('layers must be a list of positive integers')
            prev_layer = layers[l - 1] if l > 0 else nx
            weights_I.update({
                'W' + str(l + 1): np.random.randn(
                    layers[l], prev_layer) * np.sqrt(2/prev_layer),
                'b' + str(l + 1): np.zeros((layers[l], 1))})
        return weights_I

    def __init__(self, nx, layers):
        """ Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = self.Weights_init(nx, layers)

    @property
    def L(self):
        """ Layers getter """
        return self.__L

    @property
    def cache(self):
        """ Cache getter """
        return self.__cache

    @property
    def weights(self):
        """ weights getter """
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        m = X.shape[1]
        self.__cache.update({'A0': X})
        for l in range(self.__L):
            A = self.__cache.get('A' + str(l))
            Weight = self.__weights.get('W' + str(l + 1))
            Bias = self.__weights.get('b' + str(l + 1))
            Z = np.matmul(Weight, A) + Bias
            output_A = 1/(1 + np.exp(-Z))
            self.__cache.update({'A' + str(l + 1): output_A})
        return self.__cache.get('A' + str(l + 1)), self.__cache
