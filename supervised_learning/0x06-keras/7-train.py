#!/usr/bin/env python3
"""Module that trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """trains a model using mini-batch gradient descent
    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray, shape (m, classes) contains the labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        validation_data is the data to validate the model with, if not None
        early_stopping: boolean indicating whether early stopping is used
            early stopping should only be performed if validation_data exists
            early stopping should be based on validation loss
        patience is the patience used for early stopping
        learning_rate_decay:boolean indicating when learning_rate_decay apply
            learning_rate_decay only to be applied if validation_data exists
            the decay should be performed using inverse time decay
            learning rate should decay in a stepwise fashion after each epoch
            each time the learning rate updates, Keras should print a message
        alpha: is the initial learning rate
        decay_rate: is the decay rate
        verbose: boolean that determines if output should be printed
        shuffle: boolean that determines whether to shuffle batches each epoch
            Normally, it is a good idea to shuffle, but for reproducibility,
            we have chosen to set the default to False.
    Returns: the History object generated after training the model
    """

    def lr_scheduler(epoch):
        """Schedule learning rate """
        return alpha / (1 + decay_rate * epoch)
    callback_list = []
    if validation_data:
        if learning_rate_decay:
            lr_decay = K.callbacks.LearningRateScheduler(
                lr_scheduler,
                verbose=1
            )
            callback_list.append(lr_decay)
        if early_stopping:
            early_stop = K.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=patience
            )
            callback_list.append(early_stop)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callback_list,
        validation_data=validation_data,
        shuffle=shuffle,
    )
    return history
