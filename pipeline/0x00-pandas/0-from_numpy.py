#!/usr/bin/env python3
"""Module that creates a pandas dataframe from a numpy array"""
import numpy as np
import pandas as pd


def from_numpy(array):
    """"creates a pd.Dataframe from a np.ndarray
    Args:
        array is the np.ndarray from which to create the pd.Dataframe
        The columns of the pd.DataFrame should be labeled in alphabetical order
            and capitalized. There will not be more than 26 columns.
    Return: the created pdDataframe
    """
    col_size = array.shape[1]
    upper_a = 65  # First value of columns is capital A
    # create alphabetical list with size of array to name columns
    columns_list = [chr(upper_a + i) for i in range(col_size)]
    return pd.DataFrame(array, columns=columns_list)
