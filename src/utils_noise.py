"""Utilities related to noise filter."""

import matplotlib.pyplot as plt
import numpy as np

def moving_average_filter(data, window_size):
    """Moving average filter.
    """
    window = np.ones(window_size) / window_size
    smoothed_data = np.apply_along_axis(lambda x: np.convolve(x, window, mode='same'), axis=0, arr=data)
    
    for i in range(window_size // 2):
        smoothed_data[i] = data[i]
        smoothed_data[-i - 1] = data[-i - 1]
    
    return smoothed_data