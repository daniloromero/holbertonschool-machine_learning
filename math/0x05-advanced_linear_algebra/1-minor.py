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


def do_minor(matrix, i, j):
    """Removes the row and column before doing the minor matrix
    Args:
        matrix: is a list of lists
        i: row to be removed
        j: column to be removed
    Returns: Calculated minor
    """
    minor = []
    for row in matrix[:i] + matrix[i+1:]:
        minor.append(row[:j] + row[j+1:])
    return determinant(minor)


def minor(matrix):
    """calculates the minor matrix of a matrix
    Args:
        matrix: is a list of lists
    Returns: the minor matrix of matrix
    """
    if type(matrix) is not list:
        raise TypeError('matrix must be a list of lists')
    else:
        if not all(isinstance(row, list) for row in matrix):
            raise TypeError('matrix must be a list of lists')
    row_size = [len(row) for row in matrix]
    if matrix == [[]]:
    	raise ValueError("matrix must be a non-empty square matrix")
    if not all(len(matrix) == col for col in row_size):
        raise ValueError('matrix must be a square matrix')

    if len(matrix) == 1:
        return [[1]]

    if len(matrix) == 2:
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

    minor_mat = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            minor = do_minor(matrix, i, j)
            row.append(minor)
        minor_mat.append(row)
    return minor_mat
