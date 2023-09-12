"""Fits the dynamical equation using the trajectory data. This is the latest code that works with the projectile motion problem and gazebo drone data.
"""

from copy import copy
import argparse
import dill as pickle
# import pickle
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pysindy as ps
from sklearn.metrics import mean_squared_error
from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import STLSQ
from itertools import product

from river import drift

from commons import DATA_DIR, OUTPUT_DIR,THRESHOLD_MIN, THRESHOLD_MAX,NUMBER_OF_THRESHOLD_VALUES, MAX_ITERATIONS, NOISE_LEVEL, WINDOW_SIZE, MIN_WINDOW_SIZE, MAX_WINDOW_SIZE
from evaluation_metrics import AIC

def get_subset_of_data(coordinate_data, training_ratio, t):
    """
    This is a function that gets a subset of the data from the data file
    
    :param coordinate_data: The coordinates of the drone
    :param t: The time of the drone
    :param training_ratio: The ratio of the data to be used for training set vs validation set

    :return: The training and validation sets for the coordinates and time
    """
    coordinate_data_train = coordinate_data[:int(training_ratio * len(coordinate_data)), :]
    coordinate_data_validation = coordinate_data[int(training_ratio * len(coordinate_data)):, :]
    t_train = t[:int(training_ratio * len(t))]
    t_validation = t[int(training_ratio * len(t)):]
    return coordinate_data_train, coordinate_data_validation, t_train, t_validation

# Function to choose parsimonious model using AIC
def fit_and_tune_sr3(feature_library, dif_method, x_train, dt, x_valid, thresholds,
                     nus=(1 / 30, 0.1, 1 / 3, 1, 10 / 3)):
    aics = []
    params = list(product(thresholds, nus))
    for threshold, nu in params:
        model = ps.SINDy(
            optimizer=ps.SR3(threshold=threshold, nu=nu, max_iter=1000),
            feature_library=feature_library,
            differentiation_method=dif_method
        )
        try:
            model.fit(x_train, t=dt, quiet=True)

            x_dot_valid = model.differentiate(x_valid, dt)
            x_dot_pred = model.predict(x_valid)
            k = (np.abs(model.coefficients()) > 0).sum()
            aic = AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=False)
            aics.append(aic)
        except ValueError:  # SR3 sometimes fails with overflow
            aics.append(1e20)

    best_model_ix = np.argmin(np.array(aics))
    best_t, best_nu = params[best_model_ix]
    model = ps.SINDy(
        optimizer=ps.SR3(threshold=best_t, nu=best_nu, max_iter=10000),
        feature_library=feature_library,
        differentiation_method=dif_method
    )
    model.fit(x_train, t=dt, quiet=False)
    return model, best_t, best_nu

# Function to fit the best model using Weak SINDy, changed to use fit_and_tune_sr3 instead of find_lowest_rmse_threshold
def fit1(u: np.ndarray,
        t: np.ndarray) -> Tuple[ps.SINDy]:
    """
    Fits the best model using PySINDy for each coordinate (x, y, and z) separately.

    Parameters:
        u (numpy.ndarray): Array containing the coordinate and derivative data.
        t (numpy.ndarray): Array of time values corresponding to the data.

    Returns:
        tuple: A tuple containing the fitted models for x, y, and z dimensions,
        along with the corresponding coefficient arrays for each threshold value.
    """
    differentiation_method = FiniteDifference()

    polynomial_lib = ps.PolynomialLibrary(degree=1)

    threshold_scan = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, NUMBER_OF_THRESHOLD_VALUES)
    
    _, lowest_rmse_threshold, best_nu = fit_and_tune_sr3(polynomial_lib, ps.FiniteDifference(), u, t[1] - t[0], u, threshold_scan) # u_train needs to be the whole length of the data, so maybe we need a longer time period for validation

    optimizer = ps.SR3(
    threshold=lowest_rmse_threshold, nu=best_nu, thresholder="l1", max_iter=10, normalize_columns=True, tol=1e-1
    )  
    original_model = ps.SINDy(feature_library=polynomial_lib, optimizer=optimizer)
    original_model.fit(u, t=t, ensemble= True, quiet=True)
    original_model.print()

    ensemble_coefs = np.asarray(original_model.coef_list)
    median_ensemble_coefs = np.median(ensemble_coefs, axis=0) # get the median of each coefficient
    optimizer.coef_ = median_ensemble_coefs # set the coefficients to the median of the ensemble coefficients
    original_model.optimizer = optimizer # Reinitialize the optimizer with the updated coefficients

    return original_model, ensemble_coefs

# Function to find the start and end time indices 
def find_time_indices(t, start_time, window_size):
    """
    Finds the indices of time points for different intervals.

    Parameters:
        t (numpy.ndarray): Array of time values.
        start_time (float): Starting time for the interval.
        window_size (float): Size of the window for the first interval.

    Returns:
        tuple: A tuple containing two indices - start_index, end_index.
    """
    # Find the index of the first value greater than or equal to start_time
    start_index = np.searchsorted(t, start_time, side='left')
    
    # Find the index of the first value greater than end_time (exclusive)
    end_index = np.searchsorted(t, start_time + window_size, side='right')
    
    return start_index, end_index

def main() -> None:

    # Get the coordinate_data and t from projectile motion data~
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", dest="output_dir", default=OUTPUT_DIR)
    parser.add_argument("--start_time", type=float, required=True)  
    parser.add_argument("--window_size", type=float, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    start_time = args.start_time
    window_size = args.window_size

    data_file_dir = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_dir, "r") as file_read:
        coordinate_data_noise = np.array(file_read.get("coordinate_data_noise"))
        t = np.array(file_read.get("t"))

    start_index, end_index = find_time_indices(t, start_time, window_size)

    # Select the window of time to use for fitting   
    t_window = t[start_index:end_index]
    coordinate_data_noise_window = coordinate_data_noise[start_index:end_index]
    
    model_all, ensemble_coefs = fit1(coordinate_data_noise_window, t_window)

    Path(output_dir).mkdir(exist_ok=True)
    output_file_dir = Path(output_dir, "models.pkl")
    with open(output_file_dir, "wb") as file:
        pickle.dump(model_all, file)
    
    # save ensemble_coefs to a file
    output_file_dir = Path(output_dir, "ensemble_coefs.pkl")
    with open(output_file_dir, "wb") as file:
        pickle.dump(ensemble_coefs, file)


if __name__ == "__main__":
    main()