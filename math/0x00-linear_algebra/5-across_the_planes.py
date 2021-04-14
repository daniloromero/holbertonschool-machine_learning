#!/usr/bin/env python3
import numpy as np

def add_matrices2D(mat1, mat2):
    if np.shape(mat1) == np.shape(mat2):
        return np.add(mat1, mat2).tolist()
    else:
        return None
