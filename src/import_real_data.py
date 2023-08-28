# This file is to load the real data from the csv file and transform it into numpy array

from pathlib import Path
import argparse

import h5py
import numpy as np
from sklearn.metrics import mean_squared_error

from utils_graph import three_d_graph_result_new
from utils_noise import moving_average_filter

from commons import ORIGINAL_DATA_DIR, DATA_DIR, TRAJECTORY_DATA_FILE, NOISE_LEVEL, MOVING_WINDOW_SIZE

# # load the data from the csv file using genfromtxt and store it in a numpy array
# def load_data(start_time, end_time):
#     """
#     This is a function that load the data from the csv file using genfromtxt and store it in a numpy array
#     Data should be in the following format: time (in ms) , x, y, z

#     :return coordinates: The coordinates of the drone from time to time + PREDICTION_FREQUENCY
#     :return t: The time of the drone
#     """
#     data = np.genfromtxt(TRAJECTORY_DATA_FILE, delimiter=',', skip_header=1, dtype=float)
#     coordinate_data = data[:, 1:4]
#     t = data[:, 0]

#     # Find the index of the first value greater than or equal to start_time
#     start_index = np.searchsorted(t, start_time, side='left')
    
#     # Find the index of the first value greater than end_time (exclusive)
#     end_index = np.searchsorted(t, end_time, side='right')

#     coordinate_data = data[start_index:end_index, 1:4]
#     t = t[start_index:end_index]

#     return coordinate_data, t

# load the data from the csv file using genfromtxt and store it in a numpy array
def load_data():
    """
    This is a function that load the data from the csv file using genfromtxt and store it in a numpy array
    Data should be in the following format: time (in ms) , x, y, z

    :return coordinates: The coordinates of the drone
    :return t: The time of the drone
    """
    data = np.genfromtxt(TRAJECTORY_DATA_FILE, delimiter=',', skip_header=1, dtype=float)
    coordinate_data = data[:, 1:4]
    t = data[:, 0]

    return coordinate_data, t

def main()-> None:
    # logging.info("parsing drone data.")
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
    coordinate_data_noise = moving_average_filter(coordinate_data_noise, MOVING_WINDOW_SIZE)


    data_file_path = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_path, "w") as file:
        file.create_dataset(name="coordinate_data", data=coordinate_data)
        file.create_dataset(name="coordinate_data_noise", data=coordinate_data_noise)
        file.create_dataset(name="t", data=t)

    three_d_graph_result_new(coordinate_data, coordinate_data_noise, t) # check the data by plotting it 

if __name__ == '__main__':
    # logging.info("parsing drone data.")
    main()
    