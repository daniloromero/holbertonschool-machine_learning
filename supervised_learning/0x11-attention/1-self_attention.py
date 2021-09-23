#!/usr/bin/env python3
"""Creates a class SelfAttetion to calculate the attention"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Class SelfAttention"""
    def __init__(self, units):
        """SelfAttention class constructor
        units is an integer representing the number of
            hidden units in the alignment model

        W - a Dense layer with units units, to be applied to the
            previous decoder hidden state
        U - a Dense layer with units units, to be applied to the
            encoder hidden states
        V - a Dense layer with 1 units, to be applied to the
            tanh of the sum of the outputs of W and U
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1, activation='tanh')

    def call(self, s_prev, hidden_states):
        """
        s_prev is a tensor of shape (batch, units) containing
                the previous decoder hidden state
        hidden_states is a tensor of shape (batch, input_seq_len, units)
                containing the outputs of the encode
        Returns: context, weights
            context is a tensor of shape (batch, units) that contains
                    the context vector for the decoder
            weights is a tensor of shape (batch, input_seq_len, 1)
                    that contains the attention weights
        """
        prev_hs = self.W(tf.expand_dims(s_prev, 1))
        key = self.U(hidden_states)
        e = self.V(prev_hs + key)
        weights = tf.nn.softmax(e, axis=1)
        context = weights * hidden_states
        return tf.keras.backend.sum(context, axis=1), weights
