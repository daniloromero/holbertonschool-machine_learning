#!/usr/bin/env python3
""" Module that represents a binomial distribution"""


class Binomial:
    """ Class that represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """ Class construcor """
        if data is None:
            self.n = round(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.n = round(n)
            self.p = p
