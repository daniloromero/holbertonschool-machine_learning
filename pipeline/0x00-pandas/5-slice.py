#!/usr/bin/env python3
"""Script that takes every sixtyth row from a reduces dataframe of 4 columns"""
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
# Selects 4 columns and every sixtyth row
df = df[['High', 'Low', 'Close', 'Volume_(BTC)']][::60]

print(df.tail())
