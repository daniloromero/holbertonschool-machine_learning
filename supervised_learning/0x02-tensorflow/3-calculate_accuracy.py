#!/usr/bin/env python3
""" Module that calculates the accuracy of a prediction"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculates the accuracy of a prediction"""
    correct_pred = tf.argmax(y_pred, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy
