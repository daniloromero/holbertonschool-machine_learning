#!/usr/bin/env python3
""" performs matrix multiplication """


def mat_mul(mat1, mat2):
    """ function that performs matrix multiplication """
    if len(mat1[0]) != len(mat2):
        return None
    mul_mat = []
    for i, row in enumerate(mat1):
        new = []
        for j in range(len(mat2[0])):
            total = 0
            for k in range(len(mat1[0])):
                total += row[k] * mat2[k][j]
            new.append(total)
        mul_mat.append(new)
    return (mul_mat)
