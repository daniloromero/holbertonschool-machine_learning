#!/usr/bin/env python3
"""Module that calculates sensitivity for each class in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """
    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
            classes: is the number of classes
    Returns: numpy.ndarray shape (classes,) with the sensitivity of each class
    """
    classes, _ = confusion.shape
    sensitivity = np.zeros(classes)
    for i, row in enumerate(confusion):
        total = np.sum(row)
        s = row[i]/total
        np.append(sensitivity, s)
    return sensitivity
