#!/usr/bin/env python3
"""Module that creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a confusion matrix
    Args:
        labels: one-hot numpy.ndarray, shape (m, classes) contains
         correct labels for each data point
            m: is the number of data points
            classes: is the number of classes

        logits: one-hot numpy.ndarray shape (m, classes) with predicted labels
    Returns: confusion numpy.ndarray shape (classes, classes) with row indices
      showing the correct labels and column indices showing predicted labels
    """
    m, classes = labels.shape
    conf_mat = np.zeros((classes, classes))

    for i in range(m):
        m_label = labels[i]
        m_logit = logits[i]
        x, *_ = np.where(m_label == 1)
        y = np.where(m_logit == 1)
        conf_mat[x[0]][y[0]] += 1
    return conf_mat
