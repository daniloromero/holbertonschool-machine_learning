#!/usr/bin/env python3

def matrix_shape(matrix):
     """shape of a matrix"""
     shape =[]
     for item in matrix:
        shape += [len(item)]
     return shape
