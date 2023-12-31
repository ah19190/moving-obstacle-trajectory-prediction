"""Predicts a trajectory using the SINDy model. This is the old code that works with the projectile motion problem. Code does not take a moving window."""

import argparse
import pickle
from pathlib import Path
from typing import Tuple
from typing import Tuple

import h5py
import numpy as np
import pysindy as ps
from sklearn.metrics import mean_squared_error
from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import STLSQ
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from commons import DATA_DIR, OUTPUT_DIR,THRESHOLD_MIN, THRESHOLD_MAX,NUMBER_OF_THRESHOLD_VALUES, MAX_ITERATIONS, NOISE_LEVEL, WINDOW_SIZE

# Function to choose best algo hyperperameter lambda 
def find_lowest_rmse_threshold(coefs, opt, model, threshold_scan, x_test, t_test):
    dt = t_test[1] - t_test[0]
    mse = np.zeros(len(threshold_scan))
    for i in range(len(threshold_scan)):
        opt.coef_ = coefs[i]
        mse[i] = model.score(x_test, t=dt, metric=mean_squared_error)
    lowest_rmse_index = np.argmin(mse)
    print("lowest rmse index: ", lowest_rmse_index)
    return threshold_scan[lowest_rmse_index]


def fit1(u: np.ndarray,
        t: np.ndarray) -> Tuple[ps.SINDy, ps.SINDy, np.ndarray, np.ndarray]:
    """Uses PySINDy to find the equation that best fits the data u. Includes using derivatives of equations. 
       Here each coordinate x, y, and z is fitted separately.
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
        modely.fit(datay, t=t, ensemble=True)
        coefs.append(modely.coefficients())

    lowest_rmse_threshold = find_lowest_rmse_threshold(coefs, sparse_regression_optimizer, modely, threshold_scan, datay, t) 

    optimizer = STLSQ(threshold= lowest_rmse_threshold, max_iter=MAX_ITERATIONS)
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
    modelx.fit(datax, t=t, ensemble=True)
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
    modely.fit(datay, t=t, ensemble=True)
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
    modelz.fit(dataz, t=t, ensemble=True)
    modelz.print()

    ensemble_coefs_z = np.asarray(modelz.coef_list)
    median_ensemble_coefs_z = np.median(ensemble_coefs_z, axis=0) # get the median of each coefficient
    optimizer.coef_ = median_ensemble_coefs_z # set the coefficients to the median of the ensemble coefficients
    modelz.optimizer = optimizer # Reinitialize the optimizer with the updated coefficients
    # logging.info("coefficients: %s", modelz.coefficients().T)

    return (modelx, modely, modelz, xdot, ydot, zdot)

# Function to fit the best model using PySINDy
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


def main() -> None:
    # logging.info("Fitting.")

    # Get the coordinate_data and t from projectile motion data
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", dest="output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    data_file_dir = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_dir, "r") as file_read:
        coordinate_data_noise = np.array(file_read.get("coordinate_data_noise"))
        t = np.array(file_read.get("t"))

    # Select the window of time to use for fitting
    t_window = t[t <= WINDOW_SIZE]
    coordinate_data_noise_window = coordinate_data_noise[:len(t_window)]

    # (model_all, xdot, ydot, zdot) = fit2(coordinate_data_noise_window, t_window) 
    (modelx, modely, modelz, xdot, ydot, zdot) = fit1(coordinate_data_noise_window, t_window) # see fit 1 which calculates model separately for each dimension

    Path(output_dir).mkdir(exist_ok=True)
    output_file_dir = Path(output_dir, "models.pkl")
    with open(output_file_dir, "wb") as file:
        pickle.dump((modelx, modely, modelz), file)
        # pickle.dump(model_all, file)

    output_file_dir = Path(output_dir, "derivatives.hdf5")
    with h5py.File(output_file_dir, "w") as file:
        file.create_dataset(name="xdot", data=xdot)
        file.create_dataset(name="ydot", data=ydot)
        file.create_dataset(name="zdot", data=zdot) 

if __name__ == "__main__":
    main()