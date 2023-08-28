"""Predicts a trajectory using the SINDy model. This is the old code that works with the projectile motion problem."""

import warnings
from contextlib import contextmanager
from copy import copy
import argparse
import pickle
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pysindy as ps
from sklearn.metrics import mean_squared_error
from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import STLSQ
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning

from commons import DATA_DIR, OUTPUT_DIR,THRESHOLD_MIN, THRESHOLD_MAX,NUMBER_OF_THRESHOLD_VALUES, MAX_ITERATIONS, NOISE_LEVEL, WINDOW_SIZE

@contextmanager
def ignore_specific_warnings():
    filters = copy(warnings.filters)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="Sparsity parameter is too big*")
    yield
    warnings.filters = filters

# Function to choose best algo hyperparameter lambda 
def find_lowest_rmse_threshold(coefs, opt, model, threshold_scan, x_test, t_test):
    """
    Finds the threshold value that results in the lowest RMSE score for model predictions.

    Parameters:
        coefs (list): List of coefficient arrays for different threshold values.
        opt: The optimizer associated with the PySINDy model.
        model: The PySINDy model for prediction.
        threshold_scan (numpy.ndarray): Array of threshold values to consider.
        x_test (numpy.ndarray): Array containing the testing data for prediction.
        t_test (numpy.ndarray): Array of time values corresponding to the testing data.

    Returns:
        float: The threshold value that yields the lowest RMSE score for predictions.
    """
    dt = t_test[1] - t_test[0]
    mse = np.zeros(len(threshold_scan))
    mse_sim = np.zeros(len(threshold_scan))
    for i in range(len(threshold_scan)):
        opt.coef_ = coefs[i]
        mse[i] = model.score(x_test, t=dt, metric=mean_squared_error)
        x_test_sim = model.simulate(x_test[0, :], t_test, integrator="odeint")
        if np.any(x_test_sim > 1e4):
            x_test_sim = 1e4
        mse_sim[i] = np.sum((x_test - x_test_sim) ** 2)
    lowest_rmse_index = np.argmin(mse_sim) #get the lowest rmse index, important terms not truncated off 
    # print("lowest rmse index: ", lowest_rmse_index)
    return threshold_scan[lowest_rmse_index]

# Function to fit the best model using PySINDy (Current approach where each dimension is fitted separately)
def fit1(u: np.ndarray,
        t: np.ndarray) -> Tuple[ps.SINDy, ps.SINDy, np.ndarray, np.ndarray]:
    """
    Fits the best model using PySINDy for each coordinate (x, y, and z) separately.

    Parameters:
        u (numpy.ndarray): Array containing the coordinate and derivative data.
        t (numpy.ndarray): Array of time values corresponding to the data.

    Returns:
        tuple: A tuple containing the fitted models for x, y, and z dimensions,
        along with the corresponding coefficient arrays for each threshold value.
    """
    threshold_scan = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, NUMBER_OF_THRESHOLD_VALUES)
    coefs = []

    for i, threshold in enumerate(threshold_scan):
        sparse_regression_optimizer = ps.STLSQ(threshold=threshold)
        differentiation_method = FiniteDifference()
        udot = differentiation_method._differentiate(u, t)
        y = u[:, 1:2]
        ydot = udot[:, 1:2]
        datay = np.hstack((y, ydot))
        modely = ps.SINDy(optimizer=sparse_regression_optimizer,
                        differentiation_method=differentiation_method,
                        feature_names=["y", "ydot"],
                        discrete_time=False)
        modely.fit(datay, t=t, ensemble=True,quiet=True)
        coefs.append(modely.coefficients())

    
    lowest_rmse_threshold = find_lowest_rmse_threshold(coefs, sparse_regression_optimizer, modely, threshold_scan, datay, t) 

    optimizer = STLSQ(threshold= lowest_rmse_threshold, max_iter= MAX_ITERATIONS)
    differentiation_method = FiniteDifference()
    udot = differentiation_method._differentiate(u, t)
    # pylint: disable=protected-access

    # Get a model for the movement in x.
    # logging.info("Model for x")
    x = u[:, 0:1]
    xdot = udot[:, 0:1]
    datax = np.hstack((x, xdot))
    modelx = ps.SINDy(optimizer=optimizer,
                      differentiation_method=differentiation_method,
                      feature_names=["x", "xdot"],
                      discrete_time=False)
    modelx.fit(datax, t=t, ensemble=True, quiet=True)
    modelx.print()

    ensemble_coefs_x = np.asarray(modelx.coef_list)
    median_ensemble_coefs_x = np.median(ensemble_coefs_x, axis=0) # get the median of each coefficient
    optimizer.coef_ = median_ensemble_coefs_x # set the coefficients to the median of the ensemble coefficients
    modelx.optimizer = optimizer # Reinitialize the optimizer with the updated coefficients
    # logging.info("coefficients: %s", modelx.coefficients().T)

    # Get a model for the movement in y.
    # logging.info("Model for y")
    y = u[:, 1:2]
    ydot = udot[:, 1:2]
    datay = np.hstack((y, ydot))
    modely = ps.SINDy(optimizer=optimizer,
                      differentiation_method=differentiation_method,
                      feature_names=["y", "ydot"],
                      discrete_time=False)
    modely.fit(datay, t=t, ensemble=True, quiet=True)
    modely.print()

    ensemble_coefs_y = np.asarray(modely.coef_list)
    median_ensemble_coefs_y = np.median(ensemble_coefs_y, axis=0) # get the median of each coefficient
    optimizer.coef_ = median_ensemble_coefs_y # set the coefficients to the median of the ensemble coefficients
    modely.optimizer = optimizer # Reinitialize the optimizer with the updated coefficients
    # logging.info("coefficients: %s", modely.coefficients().T)

    # Get a model for the movement in z.
    # logging.info("Model for z")
    z = u[:, 2:3]
    zdot = udot[:, 2:3]
    dataz = np.hstack((z, zdot))
    modelz = ps.SINDy(optimizer=optimizer,
                      differentiation_method=differentiation_method,
                      feature_names=["z", "zdot"],
                      discrete_time=False)
    modelz.fit(dataz, t=t, ensemble=True, quiet=True)
    modelz.print()

    ensemble_coefs_z = np.asarray(modelz.coef_list)
    median_ensemble_coefs_z = np.median(ensemble_coefs_z, axis=0) # get the median of each coefficient
    optimizer.coef_ = median_ensemble_coefs_z # set the coefficients to the median of the ensemble coefficients
    modelz.optimizer = optimizer # Reinitialize the optimizer with the updated coefficients
    # logging.info("coefficients: %s", modelz.coefficients().T)

    return (modelx, modely, modelz, xdot, ydot, zdot)

# Function to fit the best model using PySINDy (old approach where we fit all dimensions at once)
def fit2(u: np.ndarray,
        t: np.ndarray) -> Tuple[ps.SINDy, np.ndarray, np.ndarray, np.ndarray]:
    """Uses PySINDy to find the equation that best fits the data u. Does not use derivatives of equations. 
    Here each coordinate x, y, and z is fitted together.
    """
    xdot = u[:, 0:1]
    ydot = u[:, 1:2]
    zdot = u[:, 2:3]
    data_all = np.hstack((u, xdot, ydot, zdot))

    threshold_scan = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, NUMBER_OF_THRESHOLD_VALUES)
    coefs = []

    for i, threshold in enumerate(threshold_scan):
        sparse_regression_optimizer = ps.STLSQ(threshold=threshold)
        differentiation_method = FiniteDifference()

        # Fit the model using pysindy
        model_all = ps.SINDy(optimizer=sparse_regression_optimizer,
                            differentiation_method=differentiation_method,
                            feature_names=["x", "xdot", "y", "ydot", "z", "zdot"],
                            discrete_time=False)
        model_all.fit(data_all, t=t, quiet=True) # ensemble here would cause the coefs to be similar for all thresholds
        coefs.append(model_all.coefficients())
    # print("coefs: ", coefs)
    lowest_rmse_threshold = find_lowest_rmse_threshold(coefs, sparse_regression_optimizer, model_all, threshold_scan, data_all, t)

    optimizer = STLSQ(threshold=lowest_rmse_threshold, max_iter=MAX_ITERATIONS)
    differentiation_method = FiniteDifference()

    # Combine the data for all dimensions (x, y, and z)
    data_all = np.hstack((u, xdot, ydot, zdot))
    # Fit the model using pysindy
    model_all = ps.SINDy(optimizer=optimizer,
                        differentiation_method=differentiation_method,
                        feature_names=["x", "xdot", "y", "ydot", "z", "zdot"],
                        discrete_time=False)
    model_all.fit(data_all, t=t, ensemble=True, quiet=True)
    # ensemble_coefs = model_all.coef_list

    ensemble_coefs = np.asarray(model_all.coef_list)
    median_ensemble_coefs = np.median(ensemble_coefs, axis=0) # get the median of each coefficient
    optimizer.coef_ = median_ensemble_coefs # set the coefficients to the median of the ensemble coefficients
    model_all.optimizer = optimizer # Reinitialize the optimizer with the updated coefficients

    model_all.print() # comment this out if you do not want the model printed to terminal 
    return (model_all, xdot, ydot, zdot)

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
    # logging.info("Fitting.")

    # Get the coordinate_data and t from projectile motion data
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", dest="output_dir", default=OUTPUT_DIR)
    parser.add_argument("--start_time", type=float, required=True)  
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    start_time = args.start_time

    data_file_dir = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_dir, "r") as file_read:
        coordinate_data_noise = np.array(file_read.get("coordinate_data_noise"))
        t = np.array(file_read.get("t"))

    start_index, end_index = find_time_indices(t, start_time, WINDOW_SIZE)
    
    # Select the window of time to use for fitting   
    t_window = t[start_index:end_index]
    coordinate_data_noise_window = coordinate_data_noise[start_index:end_index]

    with ignore_specific_warnings():
        (modelx, modely, modelz, xdot, ydot, zdot) = fit1(coordinate_data_noise_window, t_window)

    Path(output_dir).mkdir(exist_ok=True)
    output_file_dir = Path(output_dir, "models.pkl")
    with open(output_file_dir, "wb") as file:
        pickle.dump((modelx, modely, modelz), file)

    output_file_dir = Path(output_dir, "derivatives.hdf5")
    with h5py.File(output_file_dir, "w") as file:
        file.create_dataset(name="xdot", data=xdot)
        file.create_dataset(name="ydot", data=ydot)
        file.create_dataset(name="zdot", data=zdot) 

if __name__ == "__main__":
    main()