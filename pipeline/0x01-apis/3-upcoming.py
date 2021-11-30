#!/usr/bin/env python3
"""Script that print the location of a user requested to Github API"""
import requests
import time

if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches/upcoming/'
    r = requests.get(url)
    answer = r.json()
    # sort lauches by date_unix field and get lauch name and time
    sorted_launches = sorted(answer, key=lambda d: int(d['date_unix']))
    launch_name = sorted_launches[0]['name']
    launch_time = sorted_launches[0]['date_local']
    # Request rocket endpoint by id and get rocket name
    rocket_id = sorted_launches[0]['rocket']
    rocket_r = requests.get('https://api.spacexdata.com/v4/rockets/'
                            + rocket_id)
    rocket_name = rocket_r.json()['name']
    # Request launchpads endpoint and get launchpad name and locality
    launchpad_id = sorted_launches[0]['launchpad']
    launchpad_r = requests.get('https://api.spacexdata.com/v4/launchpads/'
                               + launchpad_id)
    launchpad_name = launchpad_r.json()['name']
    launchpad_locality = launchpad_r.json()['locality']

    print('{} ({}) {} - {} ({})'.format(launch_name, launch_time, rocket_name,
                                        launchpad_name, launchpad_locality))
