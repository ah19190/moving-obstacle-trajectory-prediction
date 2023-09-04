# Code to run the simulation
# Generate the data for a projectile motion problem every PREDICTION_FREQUENCY seconds 
# We will then take WINDOW_SIZE seconds of data to predict the next PREDICTION_TIME seconds of data 

import numpy as np
import argparse
import h5py
import subprocess
from pathlib import Path

from commons import DATA_DIR, PREDICTION_FREQUENCY, WINDOW_SIZE

def run_import_real_data_script():
    command = ["python3", "import_real_data.py"]
    subprocess.run(command)

def run_generate_projectile_data_script():
    command = ["python3", "generate_projectile_data.py"]
    subprocess.run(command)

def run_fit_script(time):
    command = ["python3", "_fit_cont.py", "--start_time", str(time)]
    # command = ["python3", "_fit2.py", "--start_time", str(time)]
    subprocess.run(command)

def run_predict_script(time):
    command = ["python3", "_predict_cont.py", "--start_time", str(time)]
    # command = ["python3", "_predict2.py", "--start_time", str(time)]
    subprocess.run(command)

def main(): 
    # Get the coordinate_data and t from projectile motion data
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir
    
    run_import_real_data_script() # This will get data from whichever file specified in commons.py
    # run_generate_projectile_data_script() # This will generate data for a projectile motion problem 
    
    data_file_dir = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_dir, "r") as file_read:
        t = np.array(file_read.get("t"))

    # Get start and end time for the while loop
    start_time = t[0]
    end_time = t[-1] 

    # This is the part where I fit and predict every PREDICTION_FREQUENCY seconds of data
    while start_time <= end_time - PREDICTION_FREQUENCY - WINDOW_SIZE:      

        run_fit_script(start_time)

        run_predict_script(start_time)

        start_time += PREDICTION_FREQUENCY

if __name__ == "__main__":
    main()