#!/usr/bin/env python3
"""Module that answer queations from a reference text"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """Answer question from reference text
    Args
        reference is the reference text
    Outputs an answer to stdout
    If the answer cannot be found in the reference text,
        respond with Sorry, I do not understand your question.
    """
    end_session = ['exit', 'bye', 'quit', 'goodbye']
    while True:
        question = input('Q: ')
        if question.lower() in end_session:
            print('A: Goodbye')
            break
        answer = question_answer(question, reference)
        if answer:
            print('A:' + answer)
        else:
            print('Sorry, I do not understand your question')
