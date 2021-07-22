#!/usr/bin/env python3
""" Module that calculates a determinant"""


def determinant_recur(matrix, total=0):
    """Calculates determinat od multy dimension matrix
    Args:
        matrix: is a list of lists
    Returns: the determinant of matrix
    """
    width = list(range(len(matrix)))

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    for col in width:
        copy_m = list(map(list, matrix))
        copy_m = copy_m[1:]
        height = len(copy_m)
        for row in range(height):
            copy_m[row] = copy_m[row][0:col] + copy_m[row][col+1:]
        sign = (-1) ** (col % 2)
        sub_det = determinant_recur(copy_m)
        total += sign * matrix[0][col] * sub_det

    return total


def determinant(matrix):
    """calculates the determinant of a matrix
    Args:
        matrix: is a list of lists
    Returns: the determinant of matrix
    """
    if matrix == [[]]:
        return 1

    if type(matrix) is not list:
        raise TypeError('matrix must be a list of lists')
    else:
        if not all(isinstance(row, list) for row in matrix):
            raise TypeError('matrix must be a list of lists')
    row_size = [len(row) for row in matrix]
    if not all(len(matrix) == col for col in row_size):
        raise ValueError('matrix must be a square matrix')

    if len(matrix[0]) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    return determinant_recur(matrix)
