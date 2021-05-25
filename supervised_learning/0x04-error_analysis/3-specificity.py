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
    true_neg = np.zeros(1, dtype=float)
    false_pos = np.zeros(classes, dtype=float)
    for i, row in enumerate(confusion):
        true_neg += row[i]
        false_pos[i] = np.sum(confusion[:, i]) - row[i]
    for i in range(len(false_pos)):
        specificity[i] = true_neg / (true_neg + false_pos[i])
    return specificity


