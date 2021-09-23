#!/usr/bin/env python3
"""RNNDecoder to decode for machine translation"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNNDecoder class"""

    def __init__(self, vocab, embedding, units, batch):
        """class constructor
        Args:
            vocab is an integer representing the size of the output vocabulary
            embedding is an integer representing the dimensionality of the
                    embedding vector
            units is an integer representing the number of hidden units in
                    the RNN cell
            batch is an integer representing the batch size
        """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform'
                                       )
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Args:
            x is a tensor of shape (batch, 1) containing the previous word in
                       the target sequence as an index of the target vocabulary
           s_prev is a tensor of shape (batch, units) containing the previous
                    decoder hidden state
           hidden_states is a tensor of shape (batch, input_seq_len, units)
                    containing the outputs of the encoder
        Returns: y, s
            y is a tensor of shape (batch, vocab) containing the output word
                    as a one hot vector in the target vocabulary
            s is a tensor of shape (batch, units) containing the new decoder
                    hidden state
        """
        x = self.embedding(x)
        attention = SelfAttention(self.units)
        context, weights = attention(s_prev, hidden_states)
        x = tf.concat([tf.expand_dims(context, 1), x], 2)
        outputs, hidden = self.gru(x)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        y = self.F(outputs)
        return y, hidden
