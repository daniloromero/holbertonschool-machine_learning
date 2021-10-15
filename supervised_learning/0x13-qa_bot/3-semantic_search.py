#!/usr/bin/env python3
"""Module that performs semantic search on a corpus of documents"""
import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """performs semantic search on a corpus of documents
    Args:
        corpus_path is the path to the corpus of reference documents
            on which to perform semantic search
        sentence is the sentence from which to perform semantic search
    Returns: the reference text of the document most similar to sentence
    """
    references = [sentence]
    model = hub.load('https://tfhub.dev/google/universal-sentence'
                     '-encoder-large/5')
    for file in os.listdir(corpus_path):
        if not file.endswith('.md'):
            continue
        with open(corpus_path + '/' + file, 'r', encoding='utf-8') as f:
            references.append(f.read())

    # Embedding to the content of the files
    embeddings = model(references)
    # Create a correlation matrix
    correlation = np.inner(embeddings, embeddings)
    # Best option between the sentence and all references
    closest = np.argmax(correlation[0, 1:])
    return references[closest + 1]
