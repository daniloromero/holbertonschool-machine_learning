#!/usr/bin/env python3
""" Module defines a deep neural network performing binary classification"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def __init__(self, nx, layers, activation='sig'):
        """ Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = self.Weights_init(nx, layers)
        self.__activation = activation

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

    @property
    def activation(self):
        """  getter method """
        return self.__activation

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        m = X.shape[1]
        self.__cache.update({'A0': X})
        for i in range(self.__L):
            A = self.__cache.get('A' + str(i))
            Weight = self.__weights.get('W' + str(i + 1))
            Bias = self.__weights.get('b' + str(i + 1))
            Z = np.matmul(Weight, A) + Bias
            if i == self.__L - 1:
                # Output layer with soft,ax activation function
                output_A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
            else:
                if self.__activation == 'sig':
                    output_A = 1/(1 + np.exp(-Z))
                elif self.__activation == 'tanh':
                    output_A = np.tanh(Z)
            self.__cache.update({'A' + str(i + 1): output_A})
        return self.__cache.get('A' + str(i + 1)), self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        Y_product1 = np.sum(np.multiply(Y, np.log(A)))
        #  Y_product2 = np.multiply((1 - Y), (np.log(1.0000001 - A)))
        return -(1 / m) * Y_product1

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        pred, _ = self.forward_prop(X)
        return np.where(pred < 0.5, 0, 1), self.cost(Y, pred)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        dz_prev = []
        copy_weights = self.__weights.copy()
        for n in range(self.__L, 0, -1):
            A = cache.get('A' + str(n))
            A_prev = cache.get('A' + str(n - 1))
            wx = copy_weights.get('W' + str(n + 1))
            bx = copy_weights.get('b' + str(n))
            if n == self.__L:
                dz = A - Y
            else:
                dz = np.matmul(wx.T, dz_prev) * (A * (1 - A))
            dw = np.matmul(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            dz_prev = dz
            w = copy_weights.get('W' + str(n))
            self.__weights.update({
                'W' + str(n): w - (dw * alpha),
                'b' + str(n): bx - (db * alpha)})

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Method that Trains the deep neural network """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step < 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        cost_dict = {}
        if alpha < 0:
            raise ValueError('alpha must be positive')
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if verbose is True and i % step == 0:
                print('Cost after ' + str(i) + ' iterations: '
                      + str(self.cost(Y, A)))
                cost_dict[i] = self.cost(Y, A)
            self.gradient_descent(Y, cache, alpha)
        cost_list = cost_dict.items()
        cost_list = sorted(cost_list)
        if graph:
            x, y = zip(*cost_list)
            plt.plot(x, y)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ Saves the instance object to a file in pickle format """
        if '.pkl' not in filename:
            filename += '.pkl'
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, "rb") as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError as e:
            return None
