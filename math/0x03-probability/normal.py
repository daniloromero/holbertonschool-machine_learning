#!/usr/bin/env python3
""" Module  that represents a normal distribution """


class Normal:
    """ Class  that represents a normal distribution """

    def do_stddev(self, data, mean):
        """ calculates Standard deviation"""
        total = 0
        for item in data:
            total += (item - mean) ** 2
        return total/len(data) ** (1/2)

    def __init__(self, data=None, mean=0., stddev=1.):
        """ Class constructor """
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.mean = sum(data)/len(data)
                self.stddev = self.do_stddev(data, self.mean)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean)/self.stddev
