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
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Weight1 getter """
        return self.__W1

    @property
    def b1(self):
        """Bias 2 getter """
        return self.__b1

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
        return self.__b2

    @property
    def A2(self):
        """Weight1 getter """
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1/(1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1/(1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        Y_product1 = np.multiply(Y, np.log(A))
        Y_product2 = np.multiply((1 - Y), (np.log(1.0000001 - A)))
        return -(1 / m) * (np.sum(Y_product1 + Y_product2))

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions """
        _, pred = self.forward_prop(X)
        return np.where(pred < 0.5, 0, 1), self.cost(Y, pred)
