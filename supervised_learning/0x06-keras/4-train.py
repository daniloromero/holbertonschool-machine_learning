#!/usr/BIN/ENV/ python3
"""Module that trains a model using mini-batch gradient descent"""


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """trains a model using mini-batch gradient descent
    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray, shape (m, classes) contains the labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        verbose: boolean that determines if output should be printed
        shuffle: boolean that determines whether to shuffle batches each epoch
            Normally, it is a good idea to shuffle, but for reproducibility,
            we have chosen to set the default to False.
    Returns: the History object generated after training the model
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
