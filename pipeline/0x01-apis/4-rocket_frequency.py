#!/usr/bin/env python3
"""Script that requests to Space X API to display number of launches per rocket
"""
import requests


if __name__ == '__main__':
    # request all past launches
    url = 'https://api.spacexdata.com/v4/launches/'
    r = requests.get(url)
    answer = r.json()
    # count lauches per rocket, store it into dictionary
    launch_per_rocket = {}
    for i in range(len(answer)):
        key = answer[i]['rocket']
        if key in launch_per_rocket:
            launch_per_rocket[key] += 1
        else:
            value = 1
            launch_per_rocket[key] = value
    # get rocket name
    rocket_url = 'https://api.spacexdata.com/v4/rockets/'
    for k, v in launch_per_rocket.items():
        r = requests.get(rocket_url + k)
        rocket_name = r.json()['name']
        launch_per_rocket[rocket_name] = launch_per_rocket.pop(k)
    # sort rockets by # launches in descending order
    sorted_rockets = sorted(launch_per_rocket.items(),
                            key=lambda d: d[1], reverse=True)
    for i in sorted_rockets:
        print('{}: {}'.format(i[0], i[1]))
