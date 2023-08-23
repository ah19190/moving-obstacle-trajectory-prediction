"""Predicts a trajectory using the SINDy model."""
# This will predict PREDICTION_TIME seconds of data using the model

import argparse
import logging
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
from IPython import get_ipython

from commons import DATA_DIR, OUTPUT_DIR, TIME_OF_DATA, PREDICTION_TIME, dt, WINDOW_SIZE
from utils_graph import three_d_graph_result

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12

def main() -> None:
    # logging.info("Predicting.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", dest="output_dir", default=OUTPUT_DIR)
    shell = get_ipython().__class__.__name__
    argv = [] if (shell == "ZMQInteractiveShell") else sys.argv[1:]
    args = parser.parse_args(argv)
    data_dir = args.data_dir
    output_dir = args.output_dir

    data_file_path = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_path, "r") as file_read:
        coordinate_data = np.array(file_read.get("coordinate_data"))
        t = np.array(file_read.get("t"))

    # Select the window of time that was used for fitting only
    t_fit = t[t <= WINDOW_SIZE]
    coordinate_data_fit = coordinate_data[:len(t_fit)]

    # Select the window of time that was used for fitting + PREDICTION_TIME seconds
    t_window = t[t <= WINDOW_SIZE+ PREDICTION_TIME]
    coordinate_data_window = coordinate_data[:len(t_window)]

    models_file_path = Path(output_dir, "models.pkl")
    with open(models_file_path, "rb") as file_read:
        #(model_all) = pickle.load(file_read)
        (modelx, modely, modelz) = pickle.load(file_read)

    derivatives_file_path = Path(output_dir, "derivatives.hdf5")
    with h5py.File(derivatives_file_path, "r") as file_read:
        xdot = np.array(file_read.get("xdot"))
        ydot = np.array(file_read.get("ydot"))
        zdot = np.array(file_read.get("zdot"))

    # Time points for the prediction
    # t_predict = np.arange(TIME_OF_DATA , TIME_OF_DATA + PREDICTION_TIME, dt) # predict the next PREDICTION_TIME seconds of data
    t_predict = np.arange(WINDOW_SIZE , WINDOW_SIZE + PREDICTION_TIME, t_window[-1] - t_window[-2]) # predict the next PREDICTION_TIME seconds of data
    
    # We predict each dimension separately
    u0_x = np.hstack((coordinate_data_fit[-1, 0:1], xdot[-1])) # start point is the last data point of fit data
    simulate_data_x = modelx.simulate(u0_x, t_predict)  
    simulate_data_x = simulate_data_x[:, 0:1] # we only want the value of x coordinate, not xdot

    u0_y = np.hstack((coordinate_data_fit[-1, 1:2], ydot[-1])) # start point is the last data point of fit data
    simulate_data_y = modely.simulate(u0_y, t_predict)
    simulate_data_y = simulate_data_y[:, 0:1] # we only want the value of y coordinate, not ydot

    u0_z = np.hstack((coordinate_data_fit[-1, 2:3], zdot[-1])) # start point is the last data point of fit data
    simulate_data_z = modelz.simulate(u0_z, t_predict)
    simulate_data_z = simulate_data_z[:, 0:1] # we only want the value of z coordinate, not zdot
    
    three_d_graph_result(coordinate_data_window, simulate_data_x, simulate_data_y, simulate_data_z, t_window)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
