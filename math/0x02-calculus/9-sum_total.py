#!/usr/bin/env python3
""" Module that does sigma addition """


def summation_i_squared(n):
    """ Module that does sigma addition with sqared step """
    return sum(map(lambda i: i ** 2, range(n + 1)))

