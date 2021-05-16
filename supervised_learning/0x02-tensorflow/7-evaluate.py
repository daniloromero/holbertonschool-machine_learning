#!/usr/bin/env python3
""" Module that evaluates the output of a neural network"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """  evaluates the output of a neural network
    Args:
        X is a numpy.ndarray containing the input data to evaluate
        Y is a numpy.ndarray containing the one-hot labels for X
        save_path is the location to load the model from
    Returns:  the network’s prediction, accuracy, and loss, respectively
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        test_pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        test_acc = sess.run(accuracy, feed_dict={y: Y, x: X})
        test_loss = sess.run(loss, feed_dict={y: Y, x: X})
    return test_pred, test_acc, test_loss
