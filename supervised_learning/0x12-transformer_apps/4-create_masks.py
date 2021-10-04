#!/usr/bin/env python3
"""Module that creates all masks for training/validation"""
import tensorflow as tf


def create_masks(inputs, target):
    """creates all masks for training/validation
    Args:
        inputs is a tf.Tensor of shape (batch_size, seq_len_in)
                that contains the input sentence
        target is a tf.Tensor of shape (batch_size, seq_len_out)
                that contains the target sentence
    Return: encoder_mask, combined_mask, decoder_mask

    encoder_mask tf.Tensor padding mask of shape (batch_size, 1, 1, seq_len_in)
                to be applied in the encoder
    combined_mask tf.Tensor of shape (batch_size, 1, seq_len_out, seq_len_out)
                used in the 1st attention block in the decoder to pad and mask
                future tokens in the input received by the decoder. It takes
                the maximum between a look ahead mask and the decoder target
                padding mask.
    decoder_mask is the tf.Tensor padding mask of shape
                (batch_size, 1, 1, seq_len_in)
                used in the 2nd attention block in the decoder.
    """
    # add extra dimensions to add the padding
    # to the attention logits.
    inputs = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    target = tf.cast(tf.math.equal(target, 0), tf.float32)
    # Encoder padding mask
    # (batch_size, 1, 1, seq_len)
    encoder_mask = inputs[:, tf.newaxis, tf.newaxis, :]

    # Decoder padding mask
    # (batch_size, 1, 1, seq_len)
    decoder_mask = inputs[:, tf.newaxis, tf.newaxis, :]

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    decoder_target_mask = target[:, tf.newaxis, tf.newaxis, :]
    x, y = target.shape
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((x, 1, y, y)), -1, 0)
    combined_mask = tf.maximum(decoder_target_mask, look_ahead_mask)
    return encoder_mask, combined_mask, decoder_mask
