#!/usr/bin/env python3
""" Module that creates the training operation for the network"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """creates the training operation for the network"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train
