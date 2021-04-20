#!/usr/bin/env python3
""" Module class that represents a poisson distribution """


class Poisson:
    """ Class that represents a poisson distribution """

    EULER = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ class constructor for  poisson distribution

        Args:
            data:  list of the data to be used to estimate the distribution
            lambtha: expected number of occurences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data)/len(data)

    def factorial(self, f):
        """ Calculate the factorial for an integer f """
        factorial = 1
        if int(f) >= 1:
            for i in range(1, int(f) + 1):
                factorial = factorial * i
            return factorial

    def pmf(self, k):
        """ Calculates the value of the PMF for a given number of successes
        Args:
            k: number of successes
        Returns:
            PMF of k or 0 if k is out of range.
        """
        if k < 0:
            return 0
        k = int(k)
        top = (self.EULER ** -self.lambtha) * (self.lambtha ** k)
        bottom = self.factorial(k)
        return top/bottom
