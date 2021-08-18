#!/usr/bin/env python3
"""MOdule that represents a noisless 1D Gaussian process"""
import numpy as np


class GaussianProcess():
    """GAussian Process class"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Constructor of a noiseless 1D Gaussian process
        Args:
            X_init is a numpy.ndarray of shape (t, 1) representing the inputs
                already sampled with the black-box function
            Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
                of the black-box function for each input in X_init
            t is the number of initial samples
            l is the length parameter for the kernel
            sigma_f is the standard deviation given to the output of the
                black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between 2 matrices
        Args:
            X1 is a numpy.ndarray of shape (m, 1)
            X2 is a numpy.ndarray of shape (n, 1)
            the kernel should use the Radial Basis Function (RBF)
        Returns: covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) +\
            np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """predicts mean, standard deviation of points in Gaussian Process
        Args:
            X_s: numpy.ndarray shape(s, 1) containing all points whose mean and
                standard deviation should be calculated
                s: number of sample points
        Returns: mu, sigma
            mu: numpy.ndarray shape(s,) contaning the mean for each point X_s
            sigma: numpy.ndarray shape(s,) containing the variance....
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        # Equation (7)
        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu_s = np.reshape(mu_s, -1)
        # Equation (8)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma_s = np.diagonal(cov_s)
        return mu_s, sigma_s
