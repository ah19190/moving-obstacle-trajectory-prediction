# Code to run the simulation for the csv files
# We will take WINDOW_SIZE seconds of data to predict the next PREDICTION_TIME seconds of data 

import numpy as np
import argparse
import h5py
import subprocess
from pathlib import Path

from commons import DATA_DIR, PREDICTION_FREQUENCY, WINDOW_SIZE, MAX_WINDOW_SIZE, MIN_WINDOW_SIZE
from predict import predict

def run_import_real_data_script():
    command = ["python3", "import_data.py"]
    subprocess.run(command)

def run_fit_script(time, window_size):
    # command = ["python3", "_fit_cont.py", "--start_time", str(time)]
    command = ["python3", "fit.py", "--start_time", str(time), "--window_size", str(window_size)]
    subprocess.run(command)

# def run_predict_script2(time, window_size):
#     command = ["python3", "predict.py", "--start_time", str(time), "--window_size", str(window_size)]
#     completed_process = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    
#     # Check if the process completed successfully
#     if completed_process.returncode == 0:
#         # Parse the RMSE score from the standard output
#         output_lines = completed_process.stdout.splitlines()
#         for line in output_lines:
#             if line.startswith("RMSE Score:"):
#                 rmse_score_str = line.split(":")[1].strip()
#                 rmse_score = float(rmse_score_str)
#                 return rmse_score
#     else:
#         print("Error running the script.")
#         return None

# Function to find the start and end time indices 
def find_time_indices(t, start_time, window_size):
    """
    Finds the indices of time points for different intervals.

    Parameters:
        t (numpy.ndarray): Array of time values.
        start_time (float): Starting time for the interval.
        window_size (float): Size of the window for the first interval.

    Returns:
        start_prediction_index: when we have enough data to start predicting 
    """
    
    # Find the index of the first value greater than end_time (exclusive)
    start_prediction_index = np.searchsorted(t, start_time + window_size, side='right')
    
    return start_prediction_index

# def main(): 
#     # Get the coordinate_data and t from projectile motion data
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
#     args = parser.parse_args()
#     data_dir = args.data_dir
    
#     run_import_real_data_script() # This will get data from whichever file specified in commons.py
    
#     data_file_dir = Path(data_dir, "data.hdf5")
#     with h5py.File(data_file_dir, "r") as file_read:
#         t = np.array(file_read.get("t"))

#     # Get start and end time for the while loop
#     # start_time = t[0]
#     start_time_index = find_time_indices(t, t[0], MIN_WINDOW_SIZE)
#     # start_time = t[start_time_index]
#     start_time = t[0]
#     end_time = t[-1] 

#     # declare rmse_score
#     rmse_score = 0
#     window_size = MIN_WINDOW_SIZE

#     # total_rmse_score = 0
#     # total_count = 0
#     # This is the part where I fit and predict every PREDICTION_FREQUENCY seconds of data
#     # while start_time + window_size <= end_time - PREDICTION_FREQUENCY - MAX_WINDOW_SIZE:  
#     while start_time + window_size <= end_time - PREDICTION_FREQUENCY:  
#         run_fit_script(start_time, window_size)
        
#         # rmse_score_new, simulate_data = run_predict_script2(start_time, window_size)
#         rmse_score_new, simulate_data, coordinate_data_start_to_prediction_end, coordinate_ground_truth = predict(start_time, window_size)

#         if rmse_score_new > rmse_score and window_size > MIN_WINDOW_SIZE: # if rmse_score_new is worse than rmse_score, then decrease window_size
#             window_size = 0.75 * window_size
#             rmse_score = rmse_score_new
#         elif rmse_score_new < rmse_score and window_size < MAX_WINDOW_SIZE: # if rmse_score_new is better than rmse_score, then increase window_size
#             window_size = 1.5 * window_size
#             rmse_score = rmse_score_new
#         else: # if window already at maximum or minimum, don't change window_size
#             rmse_score = rmse_score_new

#         # print("new window size: ", window_size)

#         start_time += PREDICTION_FREQUENCY
#         # window_size += PREDICTION_FREQUENCY

def main(): 
    # Get the coordinate_data and t from projectile motion data
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir
    
    run_import_real_data_script() # This will get data from whichever file specified in commons.py
    
    data_file_dir = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_dir, "r") as file_read:
        t = np.array(file_read.get("t"))

    # Get start and end time for the while loop
    # start_time = t[0]
    start_time_index = find_time_indices(t, t[0], MIN_WINDOW_SIZE)
    # start_time = t[start_time_index]
    start_time = t[0]
    end_time = t[-1] 

    # declare rmse_score
    rmse_score = 0
    window_size = MIN_WINDOW_SIZE

    # total_rmse_score = 0
    # total_count = 0
    # This is the part where I fit and predict every PREDICTION_FREQUENCY seconds of data
    # while start_time + window_size <= end_time - PREDICTION_FREQUENCY - MAX_WINDOW_SIZE:  
    while start_time + window_size <= end_time - PREDICTION_FREQUENCY:  
        run_fit_script(start_time, window_size)
        
        # rmse_score_new, simulate_data = run_predict_script2(start_time, window_size)
        rmse_score_new, simulate_data, coordinate_data_start_to_prediction_end, coordinate_ground_truth = predict(start_time, window_size)

        # if rmse_score_new > rmse_score and window_size > MIN_WINDOW_SIZE: # if rmse_score_new is worse than rmse_score, then decrease window_size
        #     window_size = 0.75 * window_size
        #     rmse_score = rmse_score_new
        # elif rmse_score_new < rmse_score and window_size < MAX_WINDOW_SIZE: # if rmse_score_new is better than rmse_score, then increase window_size
        #     window_size = 1.5 * window_size
        #     rmse_score = rmse_score_new
        # else: # if window already at maximum or minimum, don't change window_size
        #     rmse_score = rmse_score_new

        # print("new window size: ", window_size)

        # start_time += PREDICTION_FREQUENCY
        window_size += PREDICTION_FREQUENCY
    
if __name__ == "__main__":
    main()

