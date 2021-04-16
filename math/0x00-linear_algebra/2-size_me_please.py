#!/usr/bin/env python3
"""calculates the shape of a matrix"""


def len_matrix(matrix, shape):
    """recursion to get matrix lenght"""
    if isinstance(matrix, list):
        shape.append(len(matrix))
        if (len(matrix) > 0):
            return len_matrix(matrix[0], shape)
    else:
        return shape

def matrix_shape(matrix):
    """shape of a matrix"""
    shape = []
    if isinstance(matrix, list):
        return len_matrix(matrix, shape)
