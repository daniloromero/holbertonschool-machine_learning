#!/usr/bin/env python3
""" Module that does sigma addition """


def summation_i_squared(n):
    """ Module that does sigma addition with sqared step """
    if isinstance(n, int) and n >= 1:
        return sum(map(lambda i: i ** 2, range(n + 1)))
    else:
        return None
