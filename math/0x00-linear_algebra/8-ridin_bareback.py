#!/usr/bin/env python3
""" performs matrix multiplication """


def mat_mul(mat1, mat2):
    """ function that performs matrix multiplication """
    mul_mat = []
    for i in range(len(mat1)):
        new = []
        for j in range(len(mat2[0])):
            total = 0
            for k in range(len(mat1[0])):
                total += mat1[i][k] * mat2[k][j]
            new.append(total)
        mul_mat.append(new)
    return (mul_mat)
