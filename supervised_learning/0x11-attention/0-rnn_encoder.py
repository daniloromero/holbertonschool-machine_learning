#!/usr/bin/env python3
"""RNNEncoder to encode for machine translation"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNNEncoder class"""

    def __init__(self, vocab, embedding, units, batch):
        """class constructor
        Args:
            vocab is an integer representing the size of the input vocabulary
            embedding is an integer representing the dimensionality of the
                    embedding vector
            units is an integer representing the number of hidden units in
                    the RNN cell
            batch is an integer representing the batch size
        Sets the following public instance attributes:
            batch - the batch size
            units - the number of hidden units in the RNN cell
            embedding - a keras Embedding layer that converts words from
                        the vocabulary into an embedding vector
            gru - a keras GRU layer with units units
                Should return both the full sequence of outputs
                        as well as the last hidden state
                Recurrent weights should be initialized with glorot_uniform
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform'
                                       )

    def initialize_hidden_state(self):
        """Initializes the hidden states for the RNN cell to a tensor of zeros
        Returns: a tensor of shape (batch, units)containing
                the initialized hidden states
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Args:
            x is a tensor of shape (batch, input_seq_len) containing the input
                    to the encoder layer as word indices within the vocabulary
            initial is a tensor of shape (batch, units) containing the initial
                    hidden state
        Returns: outputs, hidden
            outputs is a tensor of shape (batch, input_seq_len, units)
                    containing the outputs of the encoder
            hidden is a tensor of shape (batch, units) containing the last
                    hidden state of the encoder
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
