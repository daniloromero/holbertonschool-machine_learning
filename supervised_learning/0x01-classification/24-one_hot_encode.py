#!/usr/bin/env python3
""" Module that converts a numeric label vector into a one-hot matrix"""
import numpy as np


def one_hot_encode(Y, classes):
    """ converts a numeric label vector into a one-hot matrix"""
    n = max(Y)
    Z = np.zeros(((n + 1), classes))
    for i in range(len(Y)):
        row = Y[i]
        Z[row][i] = 1
    return Z
