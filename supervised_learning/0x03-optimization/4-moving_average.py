#!/usr/bin/env python3
"""Module that calculates the weighted moving average of a data set"""


def moving_average(data, beta):
    """calculates the weighted moving average of a data
    Args:
        data: is the list of data to calculate the moving average of
        beta: is the weight used for the moving average
    Returns: a list containing the moving averages of data
    """
    epsilon = 1 - beta
    V = 0
    mv_avg = []
    for i, data_point in enumerate(data, start=1):
        bias_correction = 1 - (beta ** i)
        V = (V * beta) + (epsilon * data_point)
        mv_avg.append(V / bias_correction)
    return mv_avg
