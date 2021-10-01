#!/usr/bin/env python3
"""Modulethat creates class Dataset to load a tensorflow dataset"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """ Dataset class"""

    def __init__(self):
        """ class constructor """
        data_set, ds_info = tfds.load('ted_hrlr_translate/pt_to_en',
                                      split=['train', 'validation'],
                                      as_supervised=True,
                                      with_info=True)
        tokens = self.tokenize_dataset(data_set[0])
        self.tokenizer_pt, self.tokenizer_en = tokens
        self.data_train = data_set[0].map(self.tf_encode)
        self.data_valid = data_set[1].map(self.tf_encode)


    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for dataset
        Args:
            data is a tf.data.Dataset whose examples are formatted as
                    a tuple (pt, en)
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English sentence
        the maximum vocab size should be set to 2**15
        Returns: tokenizer_pt, tokenizer_en
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """
        STE = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = STE.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)
        tokenizer_en = STE.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Method that encodes a translation into tokens
        Args
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English sentence
        Returns: pt_tokens, en_tokens
            pt_tokens is a np.ndarray containing the Portuguese tokens
            en_tokens is a np.ndarray. containing the English tokens
        """
        vb_en = self.tokenizer_en.vocab_size
        vb_pt = self.tokenizer_pt.vocab_size

        pt_tokns = [vb_pt] + self.tokenizer_pt.encode(pt.numpy()) + [vb_pt + 1]
        en_tokns = [vb_en] + self.tokenizer_en.encode(en.numpy()) + [vb_en + 1]
        return pt_tokns, en_tokns

    def tf_encode(self, pt, en):
        """method to act as tensorflow wrapper for the encode instance method
        Args:
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English sentence

        """
        enc_pt, enc_en = tf.py_function(func=self.encode,
                                  inp=[pt, en],
                                  Tout=[tf.int64, tf.int64])
        enc_pt = tf.ensure_shape(enc_pt, None)
        enc_en = tf.ensure_shape(enc_en, None)
        return enc_pt, enc_en

