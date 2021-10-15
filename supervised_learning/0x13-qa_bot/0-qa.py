#!/usr/bin/env python3
"""MOdule that finds snippet o text within a reference document"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Args:
        question is a string containing the question to answer
        reference is a string containing the reference document
            from which to find the answer
    Returns: a string containing the answer
        If no answer is found, return None
    """
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word'
                                              '-masking-finetuned-squad')
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)
    tokens = ['CLS'] + question_tokens + ['SEP'] + reference_tokens + ['SEP']
    inpt_wd_ids = tokenizer.convert_tokens_to_ids(tokens)
    inpt_msk = [1] * len(inpt_wd_ids)
    inpt_tp_ids = [0] * (1 + len(question_tokens) + 1) +\
                  [1] * (len(reference_tokens) + 1)

    inpt_wd_ids, inpt_msk, inpt_tp_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32),
        0), (inpt_wd_ids, inpt_msk, inpt_tp_ids))
    outputs = model([inpt_wd_ids, inpt_msk, inpt_tp_ids])
    # using `[1:]` will enforce an answer. `outputs[0][0][0]`
    # is the ignored '[CLS]' token logit
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    if answer:
        return answer
    return None
