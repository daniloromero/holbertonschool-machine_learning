#!/usr/bin/env pyhon3
""" Module class that represents a poisson distribution """


class Poisson:
    """ Class that represents a poisson distribution """
    
    def __init__(self, data=None, lambtha=1.):
        """ class constructor for  poisson distribution

        Args:
            data:  list of the data to be used to estimate the distribution
            lambtha: expected number of occurences in a given time frame
        """
        if data == None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data)/len(data)
    

    def pmf(self, k):
        """ Calculates the value of the PMF for a given number of successes
        Args:
            k: number of successes
        Returns:
            PMF of k or 0 if k is out of range.
        """
        
