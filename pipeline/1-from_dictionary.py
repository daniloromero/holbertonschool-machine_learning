#!/usr/bin/env python3
""">Module that creates a pd.Dataframe from dictionary"""
import pandas as pd

data_dict = {'First': [0.0, 0.5, 1.0, 1.5],
             'Second': ['one', 'two', 'three', 'four']}

df = pd.DataFrame(data_dict, index=['A', 'B', 'C', 'D'])
