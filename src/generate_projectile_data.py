"Generate the data for a projectile motion problem."

from pathlib import Path
import argparse
import logging

import h5py
import numpy as np 

from commons import ORIGINAL_DATA_DIR, DATA_DIR, TIME_OF_DATA, PREDICTION_TIME, dt, x0, y0, z0, v0, launch_angle

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

    # Time points for the trajectory
    # t = np.arange(0, TIME_OF_DATA, dt)
    # Time points for the prediction
    # t_ground_truth = np.arange(0, TIME_OF_DATA + PREDICTION_TIME, dt) # predict the next PREDICTION_TIME seconds of data

    # Calculate the projectile motion
    # u = projectile_motion(v0, launch_angle, t).T
    # u_ground_truth = projectile_motion(v0, launch_angle, t_ground_truth).T

    # Calculate the projectile motion 
    t = np.arange(0, TIME_OF_DATA + PREDICTION_TIME, dt) # predict the next PREDICTION_TIME seconds of data
    coordinate_data = projectile_motion(v0, launch_angle, t).T

    data_file_path = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_path, "w") as file:
        file.create_dataset(name="coordinate_data", data=coordinate_data)
        file.create_dataset(name="t", data=t)
        # file.create_dataset(name="t_ground_truth", data=t_ground_truth)
        # file.create_dataset(name="u_ground_truth", data=u_ground_truth)

if __name__ == "__main__":
    main()