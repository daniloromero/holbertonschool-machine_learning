#!/usr/bin/env python3
"""Script takes input from user with prompt Q: and print A:"""
end_session = ['exit', 'quit', 'goodbye', 'bye']
while 1:
    question = input('Q: ')
    if question.lower() in end_session:
        print('A: Goodbye')
        break
    else:
        print('A: ')
