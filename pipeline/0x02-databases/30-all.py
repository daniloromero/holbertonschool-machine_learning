#!/usr/bin/env python3
"""Module that lists all documents in a collection"""
import pymongo


def list_all(mongo_collection):
    """function that lists all documents in a collection
    Args:
        mongo_collection is  the pymongo collection object
    Returns list of all documents in a collection or empty list if no document
    """
    return mongo_collection.find()
