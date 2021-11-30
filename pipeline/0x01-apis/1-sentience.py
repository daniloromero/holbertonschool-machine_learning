#!/usr/bin/env python3
"""Method that gets list od planets of all sentient species"""
import requests


def sentientPlanets():
    """Gets list of home planets of all sentient species
    Return: list of planets
    """
    url = 'https://swapi-api.hbtn.io/api/species/'
    planets = []
    species = []
    while 1:
        r = requests.get(url)
        answer = r.json()
        species += answer['results']
        # Condition to move through pagination
        if answer['next']:
            url = answer['next']
        else:
            break
    # create list with planets id
    planet_urls = []
    for sp in species:
        if sp['homeworld'] is not None:
            planet_urls.append(sp['homeworld'])
    # request planets by id and get name into a list o planets
    for planet in planet_urls:
        r = requests.get(planet)
        answer = r.json()
        planets.append(answer.get('name'))
    return planets
