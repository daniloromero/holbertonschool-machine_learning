#!/usr/bin/env python3
""" Moduel that returns the list of school having a specific topic"""
import pymongo


def schools_by_topic(mongo_collection, topic):
    """ Function that returns the list of school having a specific topic
    Args:
        mongo_collection will be the pymongo collection object
        topic (string) will be topic searched
    Returns:  returns the list of school having a specific topic
    """
    return mongo_collection.find({'topics': topic})
