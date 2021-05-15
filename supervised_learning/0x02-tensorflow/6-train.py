#!/usr/bin/env python3
""" Module  builds, trains, and saves a neural network classifier"""
import numpy as np
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """ Method builds, trains, and saves a neural network classifier
    Args:
        X_train: is a numpy.ndarray containing the training input data
        Y_train: is a numpy.ndarray containing the training labels
        X_valid: is a numpy.ndarray containing the validation input data
        Y_valid: is a numpy.ndarray containing the validation labels
        layer_sizes: list containing the number of nodes in each layer
        activations: list containing activation functions for each layer
        alpha: is the learning rate
        iterations: is the number of iterations to train over
        save_path: designates where to save the model
    """
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    for i in range(iterations):
        _, loss_value = sess.run((train_op, loss))
        print(loss_value)

    print(sess.run(y_pred))
