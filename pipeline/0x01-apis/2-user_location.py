#!/usr/bin/env python3
"""Script that print the location of a user requested to Github API"""
import requests
import sys
import time

if __name__ == '__main__':
    # make request using user input
    url = sys.argv[1]
    r = requests.get(url)
    # tracks X-RateLimit-Reset time with current time
    reset = int(r.headers['X-RateLimit-Reset'])
    now = time.time()
    minutes = reset - now
    # alternatives answer for different status codes returned by the request
    if r.status_code == 404:
        print('Not Found')
    elif r.status_code == 403:
        print('Reset in {} min '.format(minutes))

    elif r.status_code == 200:
        answer = r.json()
        print(answer['location'])
