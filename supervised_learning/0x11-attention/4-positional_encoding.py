#!/usr/bin/env python3
"""Module that calculates positional encoding for a transformer"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positional enconding for a transformer
    Args:
        max_seq_len is an integer representing the maximum sequence length
        dm is the model depth
    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing
        the positional encoding vectors
    """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / dm)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(dm)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(max_seq_len)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return sinusoid_table
