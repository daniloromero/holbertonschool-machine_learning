#!/usr/bin/env python3
"""Module that creates class Dataset to load a tensorflow dataset"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """ Dataset class"""

    def __init__(self):
        """ class constructor """
        data_set, ds_info = tfds.load('ted_hrlr_translate/pt_to_en',
                                      split=['train', 'validation'],
                                      as_supervised=True,
                                      with_info=True)
        self.data_train = data_set[0]
        self.data_valid = data_set[1]
        tokens = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt, self.tokenizer_en = tokens

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
