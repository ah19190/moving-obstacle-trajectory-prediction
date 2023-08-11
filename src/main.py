# Code to run the simulation
# Generate the data for a projectile motion problem every PREDICTION_FREQUENCY seconds 
# We will then take WINDOW_SIZE seconds of data to predict the next PREDICTION_TIME seconds of data 

import csv
import numpy as np
import time

from commons import WINDOW_SIZE, PREDICTION_FREQUENCY, PREDICTION_TIME 

import subprocess

def run_import_real_data_script():
    command = ["python3", "import_real_data.py"]
    subprocess.run(command)

def run_fit_script(use_coordinate_data=False):
    command = ["python3", "_fit.py"]
    if use_coordinate_data:
        command.append("--use_coordinate_data")
    subprocess.run(command)

def main():
    run_import_real_data_script()
    run_fit_script(use_coordinate_data=True)

if __name__ == "__main__":
    main()