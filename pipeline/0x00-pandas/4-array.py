#!/usr/bin/env python3
"""Script that selects last 10 rows from 2 columns & converts them to numpy"""
import pandas as pd
from_file = __import__('2-from_file').from_file
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

A = df[['High', 'Close']].tail(10).to_numpy()

print(A)
