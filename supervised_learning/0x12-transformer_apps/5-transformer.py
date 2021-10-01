#!/usr/bin/env python3
"""Module that creates transformer class"""
import tensorflow as tf
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positional enconding for a transformer
    Args:
        max_seq_len is an integer representing the maximum sequence length
        dm is the model depth
    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing
        the positional encoding vectors
    """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / dm)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(dm)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(max_seq_len)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return sinusoid_table


def sdp_attention(Q, K, V, mask=None):
    """Calculates the scaled dot product attention
    Args:
        Q is a tensor with its last two dimensions as (..., seq_len_q, dk)
            containing the query matrix
        K is a tensor with its last two dimensions as (..., seq_len_v, dk)
            containing the key matrix
        V is a tensor with its last two dimensions as (..., seq_len_v, dv)
            containing the value matrix
        mask is a tensor that can be broadcast into (..., seq_len_q, seq_len_v)
            containing the optional mask, or defaulted to None
            if mask is not None, multiply -1e9 to the mask and add it to
                the scaled matrix multiplication
        The preceding dimensions of Q, K, and V are the same
    Returns: output, weights
        output a tensor with its last two dimensions as (..., seq_len_q, dv)
            containing the scaled dot product attention
        weights a tensor with its last two dimensions as
            (..., seq_len_q, seq_len_v) containing the attention weights
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHeadAttention class"""

    def __init__(self, dm, h):
        """Class constructor
        Args:
            dm is an integer representing the dimensionality of the model
            h is an integer representing the number of heads
            dm is divisible by h
        Sets the following public instance attributes:
            h - the number of heads
            dm - the dimensionality of the model
            depth - the depth of each attention head
            Wq - a Dense layer with dm units, used to generate the query matrix
            Wk - a Dense layer with dm units, used to generate the key matrix
            Wv - a Dense layer with dm units, used to generate the value matrix
            linear - a Dense layer with dm units, used to generate
                     the attention output
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is
            (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Args:
            Q is a tensor of shape (batch, seq_len_q, dk) containing
                    the input to generate the query matrix
            K is a tensor of shape (batch, seq_len_v, dk) containing
                    the input to generate the key matrix
            V is a tensor of shape (batch, seq_len_v, dv) containing
                    the input to generate the value matrix
            mask is always None
        Returns: output, weights
            outputa tensor with its last two dimensions as (..., seq_len_q, dm)
                containing the scaled dot product attention
            weights a tensor with its last three dimensions as
                (..., h, seq_len_q, seq_len_v) containing the attention weights
        """
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        output, weights = sdp_attention(q, k, v, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.dm))
        output = self.linear(concat_attention)
        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """EncoderBlock class"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class constructor
        Args:
            dm - the dimensionality of the model
            h - the number of heads
            hidden - the number of hidden units in the fully connected layer
            drop_rate - the dropout rate
        Sets the following public instance attributes:
            mha - a MultiHeadAttention layer
            dense_hidden - the hidden dense layer with hidden units
                            and relu activation
            dense_output - the output dense layer with dm units
            layernorm1 - the first layer norm layer, with epsilon=1e-6
            layernorm2 - the second layer norm layer, with epsilon=1e-6
            dropout1 - the first dropout layer
            dropout2 - the second dropout layer
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Args:
            x - a tensor of shape (batch, input_seq_len, dm)
                containing the input to the encoder block
            training - a boolean to determine if the model is training
            mask - the mask to be applied for multi head attention
        Returns: a tensor of shape (batch, input_seq_len, dm)
                containing the block’s output
        """

        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)

        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderBlock(tf.keras.layers.Layer):
    """DecoderBlock class"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """class constructor
        Args:
            dm - the dimensionality of the model
            h - the number of heads
            hidden - the number of hidden units in the fully connected layer
            drop_rate - the dropout rate
        Sets the following public instance attributes:
            mha - a MultiHeadAttention layer
            mha2 - the second MultiHeadAttention layer
            dense_hidden - the hidden dense layer with hidden units
                            and relu activation
            dense_output - the output dense layer with dm units
            layernorm1 - the first layer norm layer, with epsilon=1e-6
            layernorm2 - the second layer norm layer, with epsilon=1e-6
            layernorm3 - the third layer norm layer, with epsilon=1e-6
            dropout1 - the first dropout layer
            dropout2 - the second dropout layer
            dropout3 - the third dropout layer
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Args:
             x - a tensor of shape (batch, target_seq_len, dm)
                 containing the input to the decoder block
            encoder_output - a tensor of shape (batch, input_seq_len, dm)
                             containing the output of the encoder
            training - a boolean to determine if the model is training
            look_ahead_mask - the mask to be applied to the
                              first multi head attention layer
            padding_mask - the mask to be applied to the second
                           multi head attention layer
        Returns:a tensor of shape (batch, target_seq_len, dm)
                containing the block’s output
        """
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(out1, encoder_output,
                                               encoder_output, padding_mask)

        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)

        out3 = self.layernorm3(ffn_output + out2)
        return out3


class Encoder(tf.keras.layers.Layer):
    """Encoder class"""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """ class constructor
        Args:
            N - the number of blocks in the encoder
            dm - the dimensionality of the model
            h - the number of heads
            hidden - the number of hidden units in the fully connected layer
            input_vocab - the size of the input vocabulary
            max_seq_len - the maximum sequence length possible
            drop_rate - the dropout rate
        Sets the following public instance attributes:
            N - the number of blocks in the encoder
            dm - the dimensionality of the model
            embedding - the embedding layer for the inputs
            positional_encoding - a numpy.ndarray of shape (max_seq_len, dm)
                                  containing the positional encodings
            blocks - a list of length N containing all of the EncoderBlock‘s
            dropout - dropout layer, to be applied to the positional encodings
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Args:
            x - a tensor of shape (batch, input_seq_len, dm)
                containing the input to the encoder
            training - a boolean to determine if the model is training
            mask - the mask to be applied for multi head attention
        Returns: a tensor of shape (batch, input_seq_len, dm)
                containing the encoder output
        """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x, training, mask)
        return x


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


class Transformer(tf.keras.layers.Layer):
    """  Transformer class"""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """ Class constructor
            N - the number of blocks in the encoder and decoder
            dm - the dimensionality of the model
            h - the number of heads
            hidden - the number of hidden units in the fully connected layers
            input_vocab - the size of the input vocabulary
            target_vocab - the size of the target vocabulary
            max_seq_input - the maximum sequence length possible for the input
            max_seq_target - the maximum sequence length possible for target
            drop_rate - the dropout rate
            Sets the following public instance attributes:
                encoder - the encoder layer
                decoder - the decoder layer
                linear - a final Dense layer with target_vocab units
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
            inputs - tensor (batch, input_seq_len)containing the inputs
            target - tensor (batch, target_seq_len)containing the target
            training - a boolean to determine if the model is training
            encoder_mask - the padding mask to be applied to the encoder
            look_ahead_mask - the look ahead mask to be applied to the decoder
            decoder_mask - the padding mask to be applied to the decoder
            Returns: a tensor of shape (batch, target_seq_len, target_vocab)
                containing the transformer output
        """
        encoder_out = self.encoder(inputs, training, encoder_mask)
        decoder_out = self.decoder(target, encoder_out, training,
                                   look_ahead_mask, decoder_mask
                                   )
        output = self.linear(decoder_out)

        return output
