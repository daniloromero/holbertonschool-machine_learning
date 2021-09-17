#!/usr/bin/env python3
"""Module that calculates unigram BLEU score for a sentence"""
import numpy as np


def uni_bleu(references, sentence):
    """calculates unigram BLEU score for a sentence
    Args:
        references is a list of reference translations
            each reference translation is a list of words in the translation
        sentence is a list containing the model proposed sentence
    Returns: the unigram BLEU score
    """
    sentence_l = len(sentence)
    r = np.array([len(ref) for ref in references]).min()
    word_dict_count = {w: 0 for w in sentence}
    max_count = 0
    for ref in references:
        count = 0
        ref_dict_count = {w: ref.count(w) for w in ref}
        for k in ref_dict_count.keys():
            if k in word_dict_count:
                count += 1
        if count > max_count:
            max_count = count
    P = max_count / sentence_l

    if sentence_l > r:
        BP = 1
    else:
        BP = np.exp(1-(r/sentence_l))
    bleu = BP * np.exp(np.log(P).sum() * 1)
    return bleu
