"Generate the data for a projectile motion problem."

from pathlib import Path
import argparse
import logging

import h5py
import numpy as np 
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d

import sys
sys.path.append("..")
from commons import ORIGINAL_DATA_DIR, DATA_DIR, TIME_OF_DATA, PREDICTION_TIME, NOISE_LEVEL,MOVING_WINDOW_SIZE, dt, x0, y0, z0, v0, launch_angle, SIGMA
from utils_graph import three_d_graph_result_ground_vs_noisy
from utils_noise import moving_average_filter, running_mean, guassian_filter

# Constants 
g = 9.81  # Acceleration due to gravity (m/s^2)

# Function to calculate the projectile motion (remove z to follow the example)
def projectile_motion(v0, theta_deg, t):
    theta_rad = np.radians(theta_deg)
    vx0 = v0 * np.cos(theta_rad)
    vy0 = v0 * np.sin(theta_rad)
    vz0 = v0 * np.cos(theta_rad)

    x = x0 + vx0 * t
    y = y0 + vy0 * t - 0.5 * g * t**2
    z = z0 + vz0 * t - 0.5 * g * t**2

    return np.array([x, y, z])
    # return np.array([x, y])

def main() -> None:
    # logging.info("Generating data.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--original-data_dir",
                        dest="original_data_dir",
                        default=ORIGINAL_DATA_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir

    # Calculate the projectile motion 
    t = np.arange(0, TIME_OF_DATA + PREDICTION_TIME, dt) # predict the next PREDICTION_TIME seconds of data
    coordinate_data = projectile_motion(v0, launch_angle, t).T

    # add noise to the data using rmse
    rmse = mean_squared_error(coordinate_data, np.zeros((coordinate_data).shape), squared=False)
    coordinate_data_noise = coordinate_data + np.random.normal(0, rmse * NOISE_LEVEL, coordinate_data.shape)  # Add noise
    
    # coordinate_data_noise_clean = fft_denoiser(coordinate_data_noise, 4, to_real=True)
    
    # Apply a moving average filter to denoise the data
    coordinate_data_noise = moving_average_filter(coordinate_data_noise, MOVING_WINDOW_SIZE)
    # coordinate_data_noise = gaussian_filter1d(coordinate_data_noise, sigma=SIGMA)

    data_file_path = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_path, "w") as file:
        file.create_dataset(name="coordinate_data", data=coordinate_data)
        file.create_dataset(name="coordinate_data_noise", data=coordinate_data_noise)
        file.create_dataset(name="t", data=t)

    three_d_graph_result_ground_vs_noisy(coordinate_data,coordinate_data_noise, t) # check the effect of noise filter against original data

if __name__ == "__main__":
    main()