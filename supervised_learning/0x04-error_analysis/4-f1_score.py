#!/usr/bin/env python3
"""Module that calculates the F1 score of a confusion matrix"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ calculates the F1 score of a confusion matrix
    Args:
        confusion is a confusion numpy.ndarray of shape (classes, classes)
            classes is the number of classes
    Returns: numpy.ndarray of shape (classes,) with the F1 score of each class
    """
    classes, _ = confusion.shape
    F1_score = np.zeros(classes, dtype=float)
    recall = sensitivity(confusion)
    precision1 = precision(confusion)
    F1_score = 2 * ((precision1 * recall) / (precision1 + recall))
    return F1_score
