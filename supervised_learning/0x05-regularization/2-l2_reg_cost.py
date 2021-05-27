#!/usr/bin/env python3
"""Module that calculates cost of a neural network with L2 regularization"""
import tensorflow as tf


def l2_reg_cost(cost):
    """calculates the cost of a neural network with L2 regularization
    Args:
        cost is a tensor with the cost of the network without L2 reg
    Returns: a tensor with the cost of the network accounting for L2 reg
    """
    reg_loss = tf.losses.get_regularization_losses()
    return cost + reg_loss