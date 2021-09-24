#!/usr/bin/env python3
""" Transformer Decoder """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """ Transformer Decoder class"""

    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """ class constructor
        Args:
            N - the number of blocks in the encoder and decoder
            dm - the dimensionality of the model
            h - the number of heads
            hidden - the number of hidden units in the fully connected layers
            input_vocab - the size of the input vocabulary
            target_vocab - the size of the target vocabulary
            max_seq_input - the maximum sequence length possible for the input
            max_seq_target -  maximum sequence length possible for the target
            drop_rate - the dropout rate
        Sets the following public instance attributes:

            encoder - the encoder layer
            decoder - the decoder layer
            linear - a final Dense layer with target_vocab units
         """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Args:
            inputs - a tensor of shape (batch, input_seq_len)
                    containing the inputs
            target - a tensor of shape (batch, target_seq_len)
                    containing the target
            training - a boolean to determine if the model is training
            encoder_mask - the padding mask to be applied to the encoder
            look_ahead_mask - the look ahead mask to be applied to the decoder
            decoder_mask - the padding mask to be applied to the decoder
        Returns: a tensor of shape (batch, target_seq_len, target_vocab)
                containing the transformer output
         """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        positional_encoding = self.positional_encoding
        x += positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x, encoder_output,
                               training,
                               look_ahead_mask,
                               padding_mask)

        return x
