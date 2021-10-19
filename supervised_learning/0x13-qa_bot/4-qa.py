#!/usr/bin/env python3
""" Multi-reference """
semantic_search = __import__('3-semantic_search').semantic_search
qa = __import__('0-qa').question_answer


def qa_bot(corpus_path):
    """ Multi-reference """
    bye = ['exit', 'quit', 'goodbye', 'bye']
    while True:
        question = input("Q: ")
        if question.lower() in bye:
            print("A: Goodbye")
            break
        error = "Sorry, I do not understand your question."
        reference = semantic_search(corpus_path, question)
        answer = qa(question, reference)
        print("A: {}".format(answer if answer else error))