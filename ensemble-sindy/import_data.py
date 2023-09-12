# This file is to load the real data from the csv file and transform it into numpy array

from pathlib import Path
import argparse

import h5py
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d

from utils_graph import three_d_graph_result_ground_vs_noisy
from utils_noise import moving_average_filter

from commons import ORIGINAL_DATA_DIR, DATA_DIR, TRAJECTORY_DATA_FILE, NOISE_LEVEL, MOVING_WINDOW_SIZE, SIGMA

# load the data from the csv file using genfromtxt and store it in a numpy array
def load_data():
    """
    This is a function that load the data from the csv file using genfromtxt and store it in a numpy array
    Data should be in the following format: time (in ms) , x, y, z
    Filter the data as well as Gazebo data is not always strictly increasing in time

    :return coordinates: The coordinates of the drone
    :return t: The time of the drone
    """
    data = np.genfromtxt(TRAJECTORY_DATA_FILE, delimiter=',', skip_header=1, dtype=float)
    # Initialize variables to keep track of the filtered data
    filtered_data = [data[0]]  # Add the first row to the filtered data
    prev_t = data[0, 0]

    # Iterate through the data and filter rows where 't' is not increasing
    for row in data[1:]:
        t = row[0]
        if t > prev_t:
            filtered_data.append(row)
            prev_t = t

    filtered_data = np.array(filtered_data)  # Convert the filtered data to a numpy array

    # Separate the coordinates and 't' values
    coordinate_data = filtered_data[:, 1:4]
    t = filtered_data[:, 0]

    return coordinate_data, t

def get_subset_of_data(coordinate_data, coordinate_data_noise, t, start, end):
    """
    This is a function that gets a subset of the data from the data file
    
    :param coordinate_data: The coordinates of the drone
    :param coordinate_data_noise: The coordinates of the drone with noise
    :param t: The time of the drone
    :param start: The start of the section you want to subset (ratio of the length of the data)
    :param end: The end of the section you want to subset

    :return coordinate_data_subset: The subset of the coordinates of the drone
    :return coordinate_data_noise_subset: The subset of the coordinates of the drone with noise
    :return t_subset: The subset of the time of the drone
    """
    coordinate_data_train = coordinate_data[int(start * len(t)):int(end * len(coordinate_data))]
    coordinate_data_noise_train = coordinate_data_noise[int(start * len(t)):int(end * len(coordinate_data_noise))]
    t_train = t[int(start * len(t)):int(end * len(t))]

    return coordinate_data_train, coordinate_data_noise_train, t_train

def main()-> None:
    # parse the data 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--original-data_dir",
                        dest="original_data_dir",
                        default=ORIGINAL_DATA_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir

    coordinate_data, t = load_data() 

    # add noise to the data using rmse 
    rmse = mean_squared_error(coordinate_data, np.zeros((coordinate_data).shape), squared=False)
    coordinate_data_noise = coordinate_data + np.random.normal(0, rmse * NOISE_LEVEL, coordinate_data.shape)  # Add noise

    # Apply a moving average filter to denoise the data
    # coordinate_data_noise = moving_average_filter(coordinate_data_noise, MOVING_WINDOW_SIZE)
    
    # Use gaussian filter to denoise the data
    # coordinate_data_noise = gaussian_filter1d(coordinate_data_noise, sigma=SIGMA)

    data_file_path = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_path, "w") as file:
        file.create_dataset(name="coordinate_data", data=coordinate_data)
        file.create_dataset(name="coordinate_data_noise", data=coordinate_data_noise)
        file.create_dataset(name="t", data=t)

    # Take the first 10% of the data into a new dataset for training
    # coordinate_data_train, coordinate_data_noise_train, t_train = get_subset_of_data(coordinate_data, coordinate_data_noise, t, 0.2, 0.5)
    # three_d_graph_result_ground_vs_noisy(coordinate_data_train, coordinate_data_noise_train, t_train) # check effectiveness of noise filter by plotting against actual data

    # three_d_graph_result_ground_vs_noisy(coordinate_data, coordinate_data_noise, t) # check effectiveness of noise filter by plotting against actual data

if __name__ == '__main__':
    # logging.info("parsing drone data.")
    main()
    