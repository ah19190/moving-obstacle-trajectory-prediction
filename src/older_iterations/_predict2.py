"""Predicts a trajectory using the SINDy model. This version does not include any derivative of x, y, z."""
# This will predict PREDICTION_TIME seconds of data using the model

import argparse
import logging
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
from IPython import get_ipython

from commons import DATA_DIR, OUTPUT_DIR, PREDICTION_TIME, WINDOW_SIZE
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
    parser.add_argument("--start_time", type=float, required=True)  
    shell = get_ipython().__class__.__name__
    argv = [] if (shell == "ZMQInteractiveShell") else sys.argv[1:]
    args = parser.parse_args(argv)
    data_dir = args.data_dir
    output_dir = args.output_dir
    start_time = args.start_time

    data_file_path = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_path, "r") as file_read:
        coordinate_data = np.array(file_read.get("coordinate_data"))
        t = np.array(file_read.get("t"))
    
    # Get the true data to compare it to the predicted data
    # Find the index of the first value greater than or equal to start_time
    start_index = np.searchsorted(t, start_time, side='left')
    
    # Find the index of the first value greater than end_time (exclusive)
    end_index = np.searchsorted(t, start_time + WINDOW_SIZE, side='right')
    
    # Select the window used for fitting   
    coordinate_data_fit = coordinate_data[start_index:end_index]
    t_fit = t[start_index:end_index]

    # Find the index of the first value greater than end_time + prediction time (exclusive)
    end_index_with_prediction = np.searchsorted(t, start_time + WINDOW_SIZE + PREDICTION_TIME, side='right')

    # Select the window of time that was used for fitting + PREDICTION_TIME seconds
    t_ground_truth = t[start_index:end_index_with_prediction]
    coordinate_ground_truth = coordinate_data[start_index:end_index_with_prediction] 

    models_file_path = Path(output_dir, "models.pkl")
    with open(models_file_path, "rb") as file_read:
        (modelx, modely, modelz) = pickle.load(file_read)

    # Time points for the prediction
    t_predict = t[end_index:end_index_with_prediction]# predict the next PREDICTION_TIME seconds of data
    
    # We predict each dimension separately
    u0_x = np.hstack(coordinate_data_fit[-1, 0:1]) # start point is the last data point of fit data
    simulate_data_x = modelx.simulate(u0_x, t_predict)  
    simulate_data_x = simulate_data_x[:, 0:1] # we only want the value of x coordinate, not xdot

    u0_y = np.hstack(coordinate_data_fit[-1, 1:2]) # start point is the last data point of fit data
    simulate_data_y = modely.simulate(u0_y, t_predict)


    u0_z = np.hstack(coordinate_data_fit[-1, 2:3]) # start point is the last data point of fit data
    simulate_data_z = modelz.simulate(u0_z, t_predict)
    
    three_d_graph_result(coordinate_ground_truth, simulate_data_x, simulate_data_y, simulate_data_z, t_ground_truth)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
