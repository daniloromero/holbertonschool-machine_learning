#!/usr/bin/env python3
"""Module that trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0,verbose=True, shuffle=False):
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
        verbose: boolean that determines if output should be printed
        shuffle: boolean that determines whether to shuffle batches each epoch
            Normally, it is a good idea to shuffle, but for reproducibility,
            we have chosen to set the default to False.
    Returns: the History object generated after training the model
    """
    if validation_data and early_stopping:
        early_stop = [K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience)]
    else:
        None
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=early_stop
    )
    return history
