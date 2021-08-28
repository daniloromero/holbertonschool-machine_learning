#!/usr/bin/env python3
"""Module performs Bayesian optimization on noiseless 1D Gaussian process"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """Bayesian Optimization"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """Class constructor Sets the following public instance attributes:
        f: the black-box function
        gp: an instance of the class GaussianProcess
        X_s: a numpy.ndarray of shape (ac_samples, 1) containing all
            acquisition sample points, evenly spaced between min and max
        xsi: the exploration-exploitation factor
        minimize: a bool for minimization versus maximization

        Args:
            f: is the black-box function to be optimized
            X_init: is a numpy.ndarray of shape (t, 1) representing the inputs
                already sampled with the black-box function
            Y_init: is a numpy.ndarray of shape (t, 1) representing the outputs
                of the black-box function for each input in X_init
            t: is the number of initial samples
            bounds: is a tuple of (min, max) representing the bounds of the
                space in which to look for the optimal point
            ac_samples: is the number of samples that should be analyzed
                during acquisition
            l: is the length parameter for the kernel
            sigma_f: is the standard deviation given to the output of the
                black-box function
            xsi: is the exploration-exploitation factor for acquisition
            minimize: a bool determining whether optimization should be
                performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples)
        self.X_s = self.X_s[:, np.newaxis]
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculates the next best sample location
        Uses the Expected Improvement acquisition function
        Return: X_next, EI
            X_next is a numpy.ndarray of shape (1,) representing the next best
                sample point
            EI is a numpy.ndarray of shape (ac_samples,) containing the
                expected improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)
        # mu_sample = self.p.predict(X_sample)

        # sigma = sigma.reshape(-1, 1)

        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [1]
        if self.minimize is True:
            Y_sample_opt = np.min(self.gp.Y)
            imp = Y_sample_opt - mu - self.xsi
        else:
            Y_sample_opt = np.max(self.gp.Y)
            imp = mu - Y_sample_opt - self.xsi

        with np.errstate(divide='warn'):

            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei

    def optimize(self, iterations=100):
        """optimizes the black box function
        Args:
            iterations is the maximum number of iterations to perform
            Returns: X_opt, Y_opt
                X_opt: numpy.ndarray shape (1,) representing the optimal point
                Y_opt: numpy.ndarray shape (1,) representing
                    the optimal function value
        """
        X_sample = []
        for i in range(iterations):
            X_opt, _ = self.acquisition()
            if X_opt in X_sample:
                break
            Y_opt = self.f(X_opt)
            self.gp.update(X_opt, Y_opt)
            X_sample.append(X_opt)

        if self.minimize is True:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        return self.gp.X[idx], self.gp.Y[idx]
