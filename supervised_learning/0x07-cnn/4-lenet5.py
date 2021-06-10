#!/usr/bin/env python3
""" Module that builds a modified version of the LeNet-5 architecture"""
import tensorflow as tf


def lenet5(x, y):
    """the LeNet-5 architecture
    Args:
        x: is a tf.placeholder of shape (m, 28, 28, 1) with the input images
            m is the number of images
        y is a tf.placeholder of shape (m, 10) with the one-hot labels
    Return:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization (with default hyperparameters)
        a tensor for the loss of the network
        a tensor for the accuracy of the network
    """
    init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu
    conv_l1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=(5,5),
        padding='same',
        activation=activation,
        kernel_initializer=init
    )(x)

    max_pool1 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
    )(conv_l1)

    conv_l2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=activation,
        kernel_initializer=init
    )(max_pool1)

    max_pool2 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
    )(conv_l2)

    flatten = tf.layers.Flatten()(max_pool2)

    fc_1 = tf.layers.Dense(
        units=120,
        activation=activation,
        kernel_initializer=init
    )(flatten)

    fc_2 = tf.layers.Dense(
        units=84,
        activation=activation,
        kernel_initializer=init
    )(fc_1)

    output_layer = tf.layers.Dense(
        units=10,
        kernel_initializer=init
    )(fc_2)

    y_pred = tf.nn.softmax(output_layer)

    loss = tf.losses.softmax_cross_entropy(y, output_layer)

    train_op = tf.train.AdamOptimizer().minimize(loss)

    correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return y_pred, train_op, loss, accuracy