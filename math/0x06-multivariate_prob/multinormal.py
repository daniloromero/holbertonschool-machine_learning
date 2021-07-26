#!/usr/bin/env python3
"""the class MultiNormal that represents a Multivariate Normal distribution"""
import numpy as np


class MultiNormal:
    """Multivariate Normal Dsitribution"""

    def __init__(self, data):
        """Class constructor
        Args:
            data: numpy.ndarray of shape (d, n) containing the data set
                n is the number of data points
                d id the number of dimensions in each data point
        """
        if type(data) is not np.ndarray or len(np.shape(data)) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        d, n = np.shape(data)
        if n < 2:
            raise ValueError('data must contain multiple data points')

        self.mean = np.mean(data, axis=1, keepdims=True)
        X = data - self.mean
        self.cov = np.matmul(X, X.T) / (n - 1)
