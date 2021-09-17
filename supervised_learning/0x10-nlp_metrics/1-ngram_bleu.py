#!/usr/bin/env python3
"""Module that calculates n-gram BLEU score for a sentence"""
import numpy as np


def create_ngram(sentence, n):
    """ Creates an n-gram list from sentence"""
    n_gram_list = []
    for i in range(len(sentence) - n + 1):
        n_gram = ""
        for j in range(n):
            n_gram += sentence[i + j]
            if not j + 1 == n:
                n_gram += " "
        n_gram_list.append(n_gram)
    return n_gram_list


def ngram_bleu(references, sentence, n):
    """calculates ngram BLEU score for a sentence
    Args:
        references is a list of reference translations
            each reference translation is a list of words in the translation
        sentence is a list containing the model proposed sentence
        n is the size of the n-gram to use for evaluation
    Returns: the n-gram BLEU score
    """
    sentence_l = len(sentence)
    r = np.array([len(ref) for ref in references]).min()

    n_gram = create_ngram(sentence, n)

    word_dict_count = {w: 0 for w in n_gram}
    references = [create_ngram(ref, n) for ref in references]

    max_count = 0
    for ref in references:
        count = 0
        ref_dict_count = {w: ref.count(w) for w in ref}
        for k in ref_dict_count.keys():
            if k in word_dict_count:
                count += 1
        if count > max_count:
            max_count = count
    P = max_count / len(n_gram)

    if sentence_l > r:
        BP = 1
    else:
        BP = np.exp(1-(r/sentence_l))
    bleu = BP * P
    return bleu
