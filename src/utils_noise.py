"""Utilities related to noise filter."""

import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter1d

def moving_average_filter(data, window_size):
    """Moving average filter."""
    half_window = window_size // 2
    window = np.ones(window_size) / window_size
    smoothed_data = np.apply_along_axis(lambda x: np.convolve(x, window, mode='same'), axis=0, arr=data)
    
    for i in range(half_window):
        smoothed_data[i] = data[i]
        if -i - 1 >= 0:
            smoothed_data[-i - 1] = data[-i - 1]
    
    return smoothed_data

def guassian_filter(data, sigma):
    """Guassian filter.
    """
    smoothed_data = np.apply_along_axis(lambda x: gaussian_filter1d(x, sigma=sigma), axis=0, arr=data)
    return smoothed_data