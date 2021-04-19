#!/usr/bin/env python3
""" concatenates two matrices along a specific axis """

def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenates two matrices along a specific axis """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        cat_mat = []
        for i in range(len(mat1)):
            m = mat1[i] + mat2[i]
            cat_mat.append(m)
    return cat_mat 
