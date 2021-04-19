#!/usr/bin/env python3
""" calculates the derivative of a polynomial """


def poly_derivative(poly):
    """ calculates the derivative of a polynomial """
    if isinstance(poly, list) or len(poly) == 0:
        derivative = []
        for i in range(len(poly)):
            derivative.append(poly[i] * i)
        if len(derivative) > 1:
            return derivative[1:]
        else:
            return derivative
    else:
        return None
