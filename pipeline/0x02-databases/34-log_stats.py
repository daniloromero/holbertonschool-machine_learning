#!/usr/bin/env python3
""" script that provides some stats about Nginx logs stored in MongoD"""
from pymongo import MongoClient


if __name__ == '__main__':
    client = MongoClient('mongodb://127.0.0.1:27017')
    nginx = client.logs.nginx
    logs = nginx.count_documents()
    get = nginx.count_documents({'method': 'GET'})
    post = nginx.count_documents({'method': 'POST'})
    put = nginx.count_documents({'method': 'PUT'})
    patch = nginx.count_documents({'method': 'PATCH'})
    delete = nginx.count_documents({'method': 'DELETE'})
    status = nginx.count_documents({'method': 'GET', 'path': '/status'})
    print("{} logs".format(logs))
    print("Methods:\n" +
          "\tmethod GET: {}\n".format(get) +
          "\tmethod POST: {}\n".format(post) +
          "\tmethod PUT: {}\n".format(put) +
          "\tmethod PATCH: {}\n".format(patch) +
          "\tmethod DELETE: {}".format(delete))
    print("{} status check".format(status))
