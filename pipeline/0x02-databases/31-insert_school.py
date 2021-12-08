#!/usr/bin/env python3
"""MOdule that inserts document in a collection based on kwargs"""
import pymongo


def insert_school(mongo_collection, **kwargs):
    """Function that nserts a new document in a collection
    Args:
        mongo_collection will be the pymongo collection object
        kwargs is the attributes of the document
    Returns the new _id
    """
    return (mongo_collection.insert_one(kwargs).inserted_id)
