#!/usr/bin/env python3
""" Returns the transpose of a 2D matrix """


def matrix_transpose(matrix):
    """ Returns the transpose of a 2D matrix """
    transpose =[]
    if len(matrix) > 0:
        for i in range(len(matrix[0])):
            transpose.append([])
            for item in matrix:
                transpose[-1].append(item[i])
    return transpose
