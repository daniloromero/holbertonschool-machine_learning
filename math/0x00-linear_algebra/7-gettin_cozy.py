#!/usr/bin/env python3
import numpy as np

def cat_matrices2D(mat1, mat2, axis=0):
    return np.concatenate((mat1, mat2), axis).tolist()
