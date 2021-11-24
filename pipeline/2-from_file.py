#!/usr/bin/env python3
"""Module that loads data from a file as a pd.Dataframe"""
import pandas as pd


def from_file(filename, delimiter):
    """Loads data from a filr to a pd.Dataframe
    Args:
        filename is the file to load from
        delimiter is the column separator
    Return: the loaded pd.Dataframe
    """
    return pd.read_csv(filename, delimiter)
