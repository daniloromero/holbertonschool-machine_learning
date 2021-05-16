#!/usr/bin/env python3
""" Module  builds, trains, and saves a neural network classifier"""
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

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            acc_t_val = sess.run(accuracy, feed_dict={y: Y_train, x: X_train})
            loss_t_val = sess.run(loss, feed_dict={y: Y_train, x: X_train})
            acc_v_val = sess.run(accuracy, feed_dict={y: Y_valid, x: X_valid})
            loss_v_val = sess.run(loss, feed_dict={y: Y_valid, x: X_valid})

            if i % 100 == 0:
                print('After {} iterations:'.format(i))
                print('\tTraining Cost: {}'.format(loss_t_val))
                print('\tTraining Accuracy: {}'.format(acc_t_val))
                print('\tValidation Cost: {}'.format(loss_v_val))
                print('\tValidation Accuracy: {}'.format(acc_v_val))

            sess.run(train_op, feed_dict={y: Y_train, x: X_train})
        save = saver.save(sess, save_path)
    return save
