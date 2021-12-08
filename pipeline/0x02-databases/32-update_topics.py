#!/usr/bin/env python3
"""Module  that changes all topics of a school document based on the name"""
import pymongo


def updae_topics(mongo_collection, name, topics):
    """Funtion  that changes all topics of a school document based on the name
    Args:
        mongo_collection will be the pymongo collection object
        name (string) will be the school name to update
        topics (list of strings) list of topics approached in the school
    """
    mongo_collection.update({"name": name},
                            {"$set": {"topics": topics}},
                            multi=True)
