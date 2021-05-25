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
    true_pos = confusion.diagonal()
    prediction_pos = np.sum(confusion, axis=0)
    actual_pos = np.sum(confusion, axis=1)
    false_neg = actual_pos - true_pos
    false_pos = prediction_pos - true_pos
    true_neg = np.sum(confusion) - false_pos - false_neg - true_pos

    specificity = np.divide(true_neg,  (true_neg + false_pos))

    return specificity
