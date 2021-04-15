#!/usr/bin/env python3
""" Adds two arrays element-wise """


def add_arrays(arr1, arr2):
    """ Adds two arrays element-wise """
    array_sum =[]
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            array_sum.append(arr1[i] + arr2[i])
        return array_sum
    else:
        return None
