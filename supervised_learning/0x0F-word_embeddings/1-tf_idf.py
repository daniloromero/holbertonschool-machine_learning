#!/usr/bin/env python3
"""Module tha creates s TF-IDF embedding"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """creates a TF-ID embedding
    Args:
    sentences is a list of sentences to analyze
        vocab is a list of the vocabulary words use for the analysis
            if None, all words within sentences should be used
    Returns: embeddings, features
        embeddings is a numpy.ndarray shape(s, f) containing embeddings
            s is the number of sentences in sentences
            f is the number of features analyzed
        features is a list of the features used for embeddings
    """
    if vocab is None:
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names()
    else:
        vectorizer = TfidfVectorizer(vocabulary=vocab)
        embeddings = vectorizer.fit_transform(sentences)

    return embeddings.toarray(), vocab
