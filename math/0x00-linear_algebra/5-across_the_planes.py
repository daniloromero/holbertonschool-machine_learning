#!/usr/bin/env python3
""" Adds two matrices element-wise """


def add_matrices2D(mat1, mat2):
    """ Adds two matrices element-wise """
    mat_sum = []
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        for i in range(len(mat1)):
            new = []
            for j in range(len(mat1[0])):
                new.append(mat1[i][j] + mat2[i][j])
            mat_sum.append(new)
        return mat_sum
    else:
        return None
