#!/usr/bin/env python3
""" script that provides some stats about Nginx logs stored in MongoD"""
from pymongo import MongoClient


if __name__ == '__main__':
    client = MongoClient('mongodb://127.0.0.1:27017')
    nginx_logs = client.logs.nginx
    logs = nginx_logs.find()
    get_count = nginx_logs.find({'method': 'GET'}).count()
    post_count = nginx_logs.find({'method': 'POST'}).count()
    put_count = nginx_logs.find({'method': 'PUT'}).count()
    patch_count = nginx_logs.find({'method': 'PATCH'}).count
    delete_count = nginx_logs.find({'method': 'DELETE'}).count()
    statuscount = nginx_logs.find({'method': 'GET', 'path': '/status'}).count()
    print("{} logs".format(logs.count()))
    print("Methods:\n" +
          "\tmethod GET: {}\n".format(get_count) +
          "\tmethod POST: {}\n".format(post_count) +
          "\tmethod PUT: {}\n".format(put_count) +
          "\tmethod PATCH: {}\n".format(patch_count) +
          "\tmethod DELETE: {}".format(delete_count))
    print("{} status check".format(statuscount))
