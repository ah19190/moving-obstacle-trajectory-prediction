"""Predicts a trajectory using the SINDy model. This is the old code that works with the projectile motion problem."""

import warnings
from contextlib import contextmanager
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
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

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
def find_lowest_rmse_threshold(coefs, opt, model, threshold_scan, u_test, t_test):
    """
    Finds the threshold value that results in the lowest RMSE score for model predictions.

    Parameters:
        coefs (list): List of coefficient arrays for different threshold values.
        opt: The optimizer associated with the PySINDy model.
        model: The PySINDy model for prediction.
        threshold_scan (numpy.ndarray): Array of threshold values to consider.
        u_test (numpy.ndarray): Array containing the testing data for prediction.
        t_test (numpy.ndarray): Array of time values corresponding to the testing data.

    Returns:
        float: The threshold value that yields the lowest RMSE score for predictions.
    """
    dt = t_test[1] - t_test[0]
    mse = np.zeros(len(threshold_scan))
    mse_sim = np.zeros(len(threshold_scan))
    for i in range(len(threshold_scan)):
        opt.coef_ = coefs[i]
        mse[i] = model.score(u_test, t=dt, metric=mean_squared_error)
        u_test_sim = model.simulate(u_test[0, :], t_test, integrator="odeint") #error here 
        if np.any(u_test_sim > 1e4):
            u_test_sim = 1e4
        mse_sim[i] = np.sum((u_test - u_test_sim) ** 2)
    lowest_rmse_index = np.argmin(mse_sim) #get the lowest rmse index, important terms not truncated off 
    print("lowest rmse index: ", lowest_rmse_index)
    return threshold_scan[lowest_rmse_index]

# Algorithm to scan over threshold values during Ridge Regression, and select highest performing model on the test set
def rudy_algorithm2(
    x_train,
    x_test,
    t,
    pde_lib,
    dtol,
    alpha=1e-5,
    tol_iter=25,
    normalize_columns=True,
    optimizer_max_iter=20,
    optimization="STLSQ",
):

    # Do an initial least-squares fit to get an initial guess of the coefficients
    optimizer = ps.STLSQ(
        threshold=0,
        alpha=0,
        max_iter=optimizer_max_iter,
        normalize_columns=normalize_columns,
        ridge_kw={"tol": 1e-10},
    )

    # Compute initial model
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(x_train, t=t)

    # Set the L0 penalty based on the condition number of Theta
    l0_penalty = 1e-3 * np.linalg.cond(optimizer.Theta)
    coef_best = optimizer.coef_

    # Compute MSE on the testing x_dot data (takes x_test and computes x_dot_test)
    error_best = model.score(
        x_test, metric=mean_squared_error, squared=False
    ) + l0_penalty * np.count_nonzero(coef_best)

    coef_history_ = np.zeros((coef_best.shape[0], 
                              coef_best.shape[1], 
                              1 + tol_iter))
    error_history_ = np.zeros(1 + tol_iter)
    coef_history_[:, :, 0] = coef_best
    error_history_[0] = error_best
    tol = dtol

    # Loop over threshold values, note needs some coding 
    # if not using STLSQ optimizer.
    for i in range(tol_iter):
        if optimization == "STLSQ":
            optimizer = ps.STLSQ(
                threshold=tol,
                alpha=alpha,
                max_iter=optimizer_max_iter,
                normalize_columns=normalize_columns,
                ridge_kw={"tol": 1e-10},
            )
        model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
        model.fit(x_train, t=t)
        coef_new = optimizer.coef_
        coef_history_[:, :, i + 1] = coef_new
        error_new = model.score(
            x_test, metric=mean_squared_error, squared=False
        ) + l0_penalty * np.count_nonzero(coef_new)
        error_history_[i + 1] = error_new
        
        # If error improves, set the new best coefficients
        if error_new <= error_best:
            error_best = error_new
            coef_best = coef_new
            tol += dtol
        else:
            tol = max(0, tol - 2 * dtol)
            dtol = 2 * dtol / (tol_iter - i)
            tol += dtol
    return coef_best, error_best, coef_history_, error_history_

# Function to fit the best model using PySINDy (Old approach where each dimension is fitted separately and just uses ensemble SINDy)
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

# Function to fit the best model using PySINDy (current approach: Weak SINDy)
def fit3(u: np.ndarray,
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
    udot = ps.FiniteDifference()._differentiate(u, t=t)
    
    # Define weak form ODE library
    # defaults to derivative_order = 0 if not specified,
    # and if spatial_grid is not specified, defaults to None,
    # which allows weak form ODEs.
    library_functions = [lambda x: x]
    library_function_names = [lambda x: x]
    pde_lib = ps.WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        spatiotemporal_grid=t,
        is_uniform=True,
        K=100,
    )

    threshold_scan = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, NUMBER_OF_THRESHOLD_VALUES)
    coefs = []

    for i, threshold in enumerate(threshold_scan):
        sparse_regression_optimizer = ps.STLSQ(threshold=threshold)
        # Fit the model using pysindy
        model_all = ps.SINDy(feature_library=pde_lib,
                             optimizer=sparse_regression_optimizer,
                            discrete_time=False)
        model_all.fit(u, t=t, quiet=True) # ensemble here would cause the coefs to be similar for all thresholds

        ode_lib = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        include_bias=True,
        )
        sparse_regression_optimizer = ps.STLSQ(threshold=threshold)
        original_model = ps.SINDy(feature_library=ode_lib, optimizer=sparse_regression_optimizer)
        original_model.fit(u, t=t, quiet=True)
        coefs.append(original_model.coefficients())
    
    lowest_rmse_threshold = find_lowest_rmse_threshold(coefs, sparse_regression_optimizer, original_model, threshold_scan, u, t)

    # Instantiate and fit the SINDy model with the integral of u_dot
    optimizer = ps.SR3(
    threshold=lowest_rmse_threshold, thresholder="l1", max_iter=10, normalize_columns=True, tol=1e-1
    )  
    model = ps.SINDy(feature_library=ode_lib, optimizer=optimizer)
    model.fit(u, t=t, ensemble= True, quiet=True)

    ensemble_coefs = np.asarray(model.coef_list)
    median_ensemble_coefs = np.median(ensemble_coefs, axis=0) # get the median of each coefficient
    optimizer.coef_ = median_ensemble_coefs # set the coefficients to the median of the ensemble coefficients
    model.optimizer = optimizer # Reinitialize the optimizer with the updated coefficients

    ode_lib = ps.CustomLibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    include_bias=True,
    )
    optimizer = ps.SR3(
    threshold=lowest_rmse_threshold, thresholder="l1", max_iter=10, normalize_columns=True, tol=1e-1
    )  
    original_model = ps.SINDy(feature_library=ode_lib, optimizer=optimizer)
    original_model.fit(u, t=t, ensemble= True, quiet=True)
    original_model.print()

    ensemble_coefs = np.asarray(original_model.coef_list)
    median_ensemble_coefs = np.median(ensemble_coefs, axis=0) # get the median of each coefficient
    optimizer.coef_ = median_ensemble_coefs # set the coefficients to the median of the ensemble coefficients
    original_model.optimizer = optimizer # Reinitialize the optimizer with the updated coefficients

    return original_model

# Tried out Rudy Algorithm but returned errors 
def fit4(u: np.ndarray, t: np.ndarray) -> Tuple[ps.SINDy]:
    """
    Fits the best model using PySINDy.

    Parameters:
        u (numpy.ndarray): Array containing the coordinate and derivative data.
        t (numpy.ndarray): Array of time values corresponding to the data.

    Returns:
        tuple: A tuple containing the fitted models for x, y, and z dimensions,
        along with the corresponding coefficient arrays for each threshold value.
    """
    library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
    library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y]
    ode_lib = ps.WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        spatiotemporal_grid=t,
        is_uniform=True,
        K=100,
    )

    # Split the data into training and testing sets in order (80% training, 20% testing)
    u_train, u_test, t_train, t_test = train_test_split(u, t, train_size=0.8, shuffle=False, stratify=None)

    # Determine the best threshold value using rudy_algorithm2
    best_threshold, _, _, _ = rudy_algorithm2(u_train, u_test, t_train, pde_lib=ode_lib, dtol=0.01)

    # Compute u_dot using FiniteDifference
    udot = ps.FiniteDifference()._differentiate(u, t=t)

    # Define library functions for weak form ODE
    library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
    library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y]
    ode_lib = ps.WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        spatiotemporal_grid=t,
        is_uniform=True,
        K=100,
    )

    # Use the best threshold in the optimizer
    optimizer = ps.STLSQ(
        threshold=best_threshold,
        alpha=1e-5,
        max_iter=20,
        normalize_columns=True,
        ridge_kw={"tol": 1e-10},
    )

    # Instantiate and fit the SINDy model
    model = ps.SINDy(feature_library=ode_lib, optimizer=optimizer, quiet=True)
    model.fit(u_train, t=t_train)

    # Update the optimizer with the selected threshold
    model.optimizer.threshold = best_threshold

    return model

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

    # Get the coordinate_data and t from projectile motion data~
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

    # with ignore_specific_warnings():
        # (modelx, modely, modelz, xdot, ydot, zdot) = fit1(coordinate_data_noise_window, t_window)
    model_all = fit3(coordinate_data_noise_window, t_window)

    Path(output_dir).mkdir(exist_ok=True)
    output_file_dir = Path(output_dir, "models.pkl")
    with open(output_file_dir, "wb") as file:
        pickle.dump(model_all, file)

    # Path(output_dir).mkdir(exist_ok=True)
    # output_file_dir = Path(output_dir, "models.pkl")
    # with open(output_file_dir, "wb") as file:
    #     pickle.dump((modelx, modely, modelz), file)

    # output_file_dir = Path(output_dir, "derivatives.hdf5")
    # with h5py.File(output_file_dir, "w") as file:
    #     file.create_dataset(name="xdot", data=xdot)
    #     file.create_dataset(name="ydot", data=ydot)
    #     file.create_dataset(name="zdot", data=zdot) 

if __name__ == "__main__":
    main()