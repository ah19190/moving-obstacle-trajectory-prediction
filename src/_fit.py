"""Predicts a trajectory using the SINDy model."""

import argparse
import logging
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

from commons import DATA_DIR, OUTPUT_DIR, THRESHOLD, MAX_ITERATIONS, NOISE_LEVEL

def fit1(u: np.ndarray,
        t: np.ndarray) -> Tuple[ps.SINDy, ps.SINDy, np.ndarray, np.ndarray]:
    """Uses PySINDy to find the equation that best fits the data u. Includes using derivatives of equations
    """
    optimizer = STLSQ(threshold=THRESHOLD, max_iter=MAX_ITERATIONS)
    differentiation_method = FiniteDifference()
    # pylint: disable=protected-access
    udot = differentiation_method._differentiate(u, t)

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
    # logging.info("coefficients: %s", modelz.coefficients().T)

    return (modelx, modely, modelz, xdot, ydot, zdot)

def fit2(u: np.ndarray,
        t: np.ndarray) -> Tuple[ps.SINDy, np.ndarray, np.ndarray, np.ndarray]:
    """Uses PySINDy to find the equation that best fits the data u. Does not use derivatives of equations. 
    """
    optimizer = STLSQ(threshold=THRESHOLD, max_iter=MAX_ITERATIONS)
    differentiation_method = FiniteDifference()
    
    xdot = u[:, 0:1]
    ydot = u[:, 1:2]
    zdot = u[:, 2:3]

    # Combine the data for all dimensions (x, y, and z)
    data_all = np.hstack((u, xdot, ydot, zdot))
    # Fit the model using pysindy
    model_all = ps.SINDy(optimizer=optimizer,
                        differentiation_method=differentiation_method,
                        feature_names=["x", "xdot", "y", "ydot", "z", "zdot"],
                        discrete_time=False)
    model_all.fit(data_all, t=t, ensemble=True)
    model_all.print() # comment this out if you do not want the model printed to terminal 

    return (model_all, xdot, ydot, zdot)


def main() -> None:
    # logging.info("Fitting.")

    # Get the u and t from projectile motion data
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", dest="output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    data_file_dir = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_dir, "r") as file_read:
        u = np.array(file_read.get("u"))
        t = np.array(file_read.get("t"))

    # add noise to the data using rmse
    rmse = mean_squared_error(u, np.zeros((u).shape), squared=False)
    u_noise = u + np.random.normal(0, rmse * NOISE_LEVEL, u.shape)  # Add noise

    (model_all, xdot, ydot, zdot) = fit2(u_noise, t) # base case we are using fit 2 for now

    Path(output_dir).mkdir(exist_ok=True)
    output_file_dir = Path(output_dir, "models.pkl")
    with open(output_file_dir, "wb") as file:
        # pickle.dump((modelx, modely), file)
        pickle.dump(model_all, file)

    output_file_dir = Path(output_dir, "derivatives.hdf5")
    with h5py.File(output_file_dir, "w") as file:
        file.create_dataset(name="xdot", data=xdot)
        file.create_dataset(name="ydot", data=ydot)
        file.create_dataset(name="zdot", data=zdot) 

if __name__ == "__main__":
    main()