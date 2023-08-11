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

from commons import DATA_DIR, OUTPUT_DIR, TIME_OF_DATA, PREDICTION_TIME, dt
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
        u = np.array(file_read.get("u"))
        t = np.array(file_read.get("t"))
        u_ground_truth = np.array(file_read.get("u_ground_truth"))
        t_ground_truth = np.array(file_read.get("t_ground_truth"))
    
    models_file_path = Path(output_dir, "models.pkl")
    with open(models_file_path, "rb") as file_read:
        (model_all) = pickle.load(file_read)

    derivatives_file_path = Path(output_dir, "derivatives.hdf5")
    with h5py.File(derivatives_file_path, "r") as file_read:
        xdot = np.array(file_read.get("xdot"))
        ydot = np.array(file_read.get("ydot"))
        zdot = np.array(file_read.get("zdot"))

    # Time points for the prediction
    t_predict = np.arange(TIME_OF_DATA , TIME_OF_DATA + PREDICTION_TIME, dt) # predict the next PREDICTION_TIME seconds of data

    # Predict the trajectory of the ball using the model
    u0_all = np.hstack((u[-1], xdot[-1], ydot[-1], zdot[-1])) #use u here instead of u_noise as we want it to start from same starting point
    u_approximation_all = model_all.simulate(u0_all, t_predict)

    # Reshape the predictions into separate arrays for each dimension
    u_approximation_x = u_approximation_all[:, :1]
    u_approximation_y = u_approximation_all[:, 1:2]
    u_approximation_z = u_approximation_all[:, 2:3]

    # graph_result(u_ground_truth, u_approximation_x, u_approximation_y, u_approximation_z, t_ground_truth)

    three_d_graph_result(u_ground_truth, u_approximation_x, u_approximation_y, u_approximation_z, t_ground_truth)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
