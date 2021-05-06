#!/usr/bin/env python3
""" class Neuron, defines a single neuron performing binary classification"""
import numpy as np
import matplotlib.pyplot as plt


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
        dw = (1 / m) * np.matmul(dz, X.T)
        db = (1 / m) * np.sum(dz)
        self.__W = self.__W - np.multiply(dw, alpha)
        self.__b = self.__b - np.multiply(db, alpha)
        return self.__W, self.__b
    
    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ method to Train the neuron """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if verbose is True  or graph is True:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step < 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        cost_dict = {} 
        for i in range(iterations + 1):
            self.forward_prop(X)
            if verbose == True and i % step == 0:
                print('Cost after ' + str(i) + ' iterations: ' + str(self.cost(Y, self.__A)))
                cost_dict[i] = self.cost(Y, self.__A)

            self.gradient_descent(X, Y, self.__A, alpha)
        cost_list = cost_dict.items()
        cost_list = sorted(cost_list)
        x, y = zip(*cost_list)
        plt.plot(x,y)
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.title('Training Cost')
        plt.show()
        return self.evaluate(X, Y)

