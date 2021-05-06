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
        """ Activated output value=> prediction of the neuron """
        return self.__A

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        Y_product1 = np.multiply(Y, np.log(A))
        Y_product2 = np.multiply((1 - Y), (np.log(1.0000001 - A)))
        return -(1 / m) * (np.sum(Y_product1 + Y_product2))

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        pred = self.forward_prop(X)
        return np.where(pred < 0.5, 0, 1), self.cost(Y, pred)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dz = A - Y
        dw = np.matmul(dz, X.T) * (1 / m)
        db = (1 / m) * np.sum(dz)
        self.__W = self.__W - np.multiply(dw, alpha)
        self.__b = self.__b - np.multiply(alpha, db)
        return self.__W, self.__b
