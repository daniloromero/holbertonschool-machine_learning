#!/usr/bin/env python3
"""Module calculates the precision for each class in a confusion matrix"""
import numpy as np


def precision(confusion):
    """calculates the precision for each class in a confusion matrix
    Args:
        confusion is a confusion numpy.ndarray of shape (classes, classes)
    Returns: numpy.ndarray, shape (classes,) with the precision of each class
    """
    classes, _ = confusion.shape
    precision = np.zeros(classes, dtype=float)
    for i, row in enumerate(confusion):
        total = np.sum(confusion[:, i])
        p = row[i] / total
        precision[i] = p
    return precision
