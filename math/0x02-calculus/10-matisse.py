#!/usr/bin/env python3
""" calculates the derivative of a polynomial """


def poly_derivative(poly):
    """ calculates the derivative of a polynomial """
    if isinstance(poly, list) or len(poly) != 0:
        derivative = []
        for i in range(1, len(poly)):
            derivative.append(poly[i] * i)
        if derivative == []:
            return [0]
        else:
            return derivative
    else:
        return None
