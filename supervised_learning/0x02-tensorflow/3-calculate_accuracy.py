#!/usr/bin/env python3
""" Module that calculates the accuracy of a prediction"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculates the accuracy of a prediction
    Args:
        y: is a placeholder for the labels of the input data
        y_pred: is a tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction
    """
    correct_pred = tf.argmax(y_pred, y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy
