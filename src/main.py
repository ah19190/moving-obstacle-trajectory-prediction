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

def run_fit_script(time):
    command = ["python3", "_fit_cont.py", "--time", str(time)]
    subprocess.run(command)

def run_predict_script(time):
    command = ["python3", "_predict_cont.py", "--time", str(time)]
    subprocess.run(command)

# Old main where I only predict once and then stop 
# def main(): 

#     run_import_real_data_script() # This will get data from whichever file specified in commons.py

#     # Get the coordinate_data and t from projectile motion data
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
#     args = parser.parse_args()
#     data_dir = args.data_dir

#     data_file_dir = Path(data_dir, "data.hdf5")
#     with h5py.File(data_file_dir, "r") as file_read:
#         t = np.array(file_read.get("t"))

#     command = ["python3", "_fit.py"]
#     subprocess.run(command)

#     command = ["python3", "_predict.py"]
#     subprocess.run(command)

#     # This part is for when i want to do continuous data generation
#     # # Get the max time of t
#     # max_time = int(t[-1])

#     # # Run the fit script for every PREDICTION_FREQUENCY seconds of data
#     # for i in range(0, max_time, PREDICTION_FREQUENCY):
#     #     run_fit_script(i)
#     #     # run_predict_script(i)

def main():
    data_buffer = []

    # Get the coordinate_data and t from projectile motion data
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir
    
    run_import_real_data_script() # This will get data from whichever file specified in commons.py
    
    data_file_dir = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_dir, "r") as file_read:
        
        t = np.array(file_read.get("t"))

    start_time = t[0]
    end_time = t[-1] 

    # This is the part where I fit and predict every PREDICTION_FREQUENCY seconds of data
    while start_time <= end_time:
        
        
        # Find the index of the first value greater than or equal to start_time
        start_index = np.searchsorted(t, start_time, side='left')
        
        # Find the index of the first value greater than end_time (exclusive)
        end_index = np.searchsorted(t, end_time, side='right')
        
        coordinate_data = data[start_index:end_index, 1:4]
        t = t[start_index:end_index]
        
        
        command = ["python3", "_fit.py"]
        subprocess.run(command)

        command = ["python3", "_predict.py"]
        subprocess.run(command)

if __name__ == "__main__":
    main()