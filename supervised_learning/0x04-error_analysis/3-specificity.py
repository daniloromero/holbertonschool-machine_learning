#!/usr/bin/env python3
"""Module calculates the specificity for each class in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """calculates the specificity for each class in a confusion matrix
    Args:
        confusion is a confusion numpy.ndarray of shape (classes, classes)
            classes is the number of classes
    Returns: numpy.ndarray of shape (classes,) with specificity of each class
    """
    classes, _ = confusion.shape
    specificity = np.zeros(classes, dtype=float)

    for i, row in enumerate(confusion):
        false_pos = np.sum(confusion[i]) - row[i]
        false_neg = np.sum(confusion[:, i]) - row[i]

        sub_total = np.sum(confusion) - false_pos - false_neg - row[i]
        specificity[i] = np.divide(sub_total,  (sub_total + false_pos))

    return specificity
