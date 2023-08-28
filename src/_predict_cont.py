"""Predicts a trajectory using the SINDy model."""
# This will predict PREDICTION_TIME seconds of data using the model

import warnings
from contextlib import contextmanager
from copy import copy
import argparse
import logging
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
from IPython import get_ipython

from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning

from commons import DATA_DIR, OUTPUT_DIR, PREDICTION_TIME, WINDOW_SIZE
from utils_graph import three_d_graph_result

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12

@contextmanager
def ignore_specific_warnings():
    filters = copy(warnings.filters)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="lsoda--  warning")
    warnings.filterwarnings("ignore", message="Sparsity parameter is too big*")
    yield
    warnings.filters = filters

def load_data(data_dir):
    """
    Loads coordinate data and time values from a data file.

    Parameters:
        data_dir (str): Directory containing the data file.

    Returns:
        tuple: A tuple containing two numpy arrays - coordinate_data and t.
            - coordinate_data: Array of coordinate data.
            - t: Array of time values corresponding to the coordinate data.
    """
    data_file_path = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_path, "r") as file_read:
        coordinate_data = np.array(file_read.get("coordinate_data"))
        t = np.array(file_read.get("t"))
    return coordinate_data, t

def load_models(output_dir):
    """
    Loads models for different dimensions from a models file.

    Parameters:
        output_dir (str): Directory containing the models file.

    Returns:
        tuple: A tuple containing three model instances - modelx, modely, and modelz.
    """
    models_file_path = Path(output_dir, "models.pkl")
    with open(models_file_path, "rb") as file_read:
        modelx, modely, modelz = pickle.load(file_read)
    return modelx, modely, modelz

def load_derivatives(output_dir):
    """
    Loads derivative data from a derivatives file.

    Parameters:
        output_dir (str): Directory containing the derivatives file.

    Returns:
        tuple: A tuple containing three numpy arrays - xdot, ydot, and zdot.
    """
    derivatives_file_path = Path(output_dir, "derivatives.hdf5")
    with h5py.File(derivatives_file_path, "r") as file_read:
        xdot = np.array(file_read.get("xdot"))
        ydot = np.array(file_read.get("ydot"))
        zdot = np.array(file_read.get("zdot"))
    return xdot, ydot, zdot

def predict_dimension(model, coordinate_data_fit, derivative_data, t_predict, dim_idx):
    """
    Predicts the trajectory of a specific dimension using a given model.

    Parameters:
        model (model): The model to use for prediction (e.g., modelx, modely, modelz).
        coordinate_data_fit (numpy.ndarray): Array containing coordinate data used for fitting.
        derivative_data (numpy.ndarray): Array containing derivative data.
        t_predict (numpy.ndarray): Array of time points for prediction.
        dim_idx (int): Index of the dimension to predict (0 for x, 1 for y, 2 for z).

    Returns:
        numpy.ndarray: Predicted trajectory data for the specified dimension.
    """
    u0 = np.hstack((coordinate_data_fit[-1, dim_idx:dim_idx + 1], derivative_data[-1])) # start point is the last data point of fit data
    with ignore_specific_warnings():
        simulate_data = model.simulate(u0, t_predict)
    return simulate_data[:, 0:1] # we only want the value of coordinate, not the derivative

def find_time_indices(t, start_time, window_size, prediction_time):
    """
    Finds the indices of time points for different intervals.

    Parameters:
        t (numpy.ndarray): Array of time values.
        start_time (float): Starting time for the interval.
        window_size (float): Size of the window for the first interval.
        prediction_time (float): Time duration for prediction.

    Returns:
        tuple: A tuple containing three indices - start_index, end_index, and end_index_with_prediction.
    """
    # Find the index of the first value greater than or equal to start_time
    start_index = np.searchsorted(t, start_time, side='left')
    
    # Find the index of the first value greater than end_time (exclusive)
    end_index = np.searchsorted(t, start_time + window_size, side='right')

    # Find the index of the first value greater than end_time + prediction time (exclusive)
    end_index_with_prediction = np.searchsorted(t, start_time + window_size + prediction_time, side='right') # There might be a rounding error here, as it is not t_fit + PREDICTION_TIME
    
    return start_index, end_index, end_index_with_prediction

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

    # Load the data from the data_dir
    coordinate_data, t = load_data(data_dir)
    
    modelx, modely, modelz = load_models(output_dir)

    xdot, ydot, zdot = load_derivatives(output_dir)

    # Get the true data to compare it to the predicted data
    start_index, end_index, end_index_with_prediction = find_time_indices(t, start_time, WINDOW_SIZE, PREDICTION_TIME)
    
    # Select the window used for fitting   
    coordinate_data_fit = coordinate_data[start_index:end_index]
    t_fit = t[start_index:end_index]

    # Select the window of time that was used for fitting + PREDICTION_TIME seconds
    t_ground_truth = t[start_index:end_index_with_prediction]
    coordinate_ground_truth = coordinate_data[start_index:end_index_with_prediction] 

    # Time points for the prediction
    t_predict = t[end_index:end_index_with_prediction] # predict the next PREDICTION_TIME seconds of data
    
    # We predict each dimension separately
    simulate_data_x = predict_dimension(modelx, coordinate_data_fit, xdot, t_predict, 0)
    simulate_data_y = predict_dimension(modely, coordinate_data_fit, ydot, t_predict, 1)
    simulate_data_z = predict_dimension(modelz, coordinate_data_fit, zdot, t_predict, 2)
    
    # Plot the simulation against the ground truth
    three_d_graph_result(coordinate_data[0: end_index_with_prediction], coordinate_ground_truth, simulate_data_x, simulate_data_y, simulate_data_z, t_ground_truth)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
