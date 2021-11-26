#!/usr/bin/env python3
"""Script that print the location of a user requested to Github API"""
import requests
import sys
import datetime as dt

if __name__ == '__main__':
    # make request using user input
    url = sys.argv[1]
    r = requests.get(url)

    # converts X-RateLimit-Reset time to minutes
    miliseconds = r.headers['X-RateLimit-Reset']
    minutes = dt.datetime.strptime(miliseconds[:-8], '%M')
    minutes = minutes.strftime('%M')
    # alternatives answer for different status codes returned by the request
    if r.status_code == 404:
        print('Not Found')
    elif r.status_code == 403:
        print('Reset in {} min '.format(minutes))

    elif r.status_code == 200:
        answer = r.json()
        print(answer['location'])
