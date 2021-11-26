#!/usr/bin/env python3
"""method that returns list of ships that can hold a given number of passengers
"""
import requests


def availableShips(passengerCount):
    """List ships that can hold a given number of passengers
    Args:
        passengerCount is the number of passenger tha the ship must hold
    Return: List of ship that hold the passengerCount
    """
    url = 'https://swapi-api.hbtn.io/api/starships/'
    spaceships = []
    while 1:
        r = requests.get(url)
        answer = r.json()
        spaceships += answer['results']
        # Condition to move through pagination
        if answer['next']:
            url = answer['next']
        else:
            break
    available_ships = []
    for ship in spaceships:
        try:
            if int(ship.get('passengers').replace(',', '')) >= passengerCount:
                available_ships.append(ship.get('name'))
        except Exception as e:  # There are spaceship with no passengers value
            continue
    return available_ships
