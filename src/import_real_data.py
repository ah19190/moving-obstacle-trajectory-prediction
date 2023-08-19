# This file is to load the real data from the csv file and transform it into numpy array

from pathlib import Path
import argparse
import logging

import h5py
import numpy as np

from commons import ORIGINAL_DATA_DIR, DATA_DIR, TRAJECTORY_DATA_FILE

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

    data_file_path = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_path, "w") as file:
        file.create_dataset(name="coordinate_data", data=coordinate_data)
        file.create_dataset(name="t", data=t) 

if __name__ == '__main__':
    # logging.info("parsing drone data.")
    main()
    