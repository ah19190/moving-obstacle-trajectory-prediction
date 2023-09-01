"Constants and common code."
from pathlib import Path

# Directory names
ORIGINAL_DATA_DIR = str(Path(Path(__file__).parent.parent, "original-data"))
DATA_DIR = str(Path(Path(__file__).parent.parent, "data"))
OUTPUT_DIR = str(Path(Path(__file__).parent.parent, "output"))

# Data file we want to parse for trajectory data 
# TRAJECTORY_DATA_FILE = "../original-data/drone_trajectory_data.csv"
TRAJECTORY_DATA_FILE = "../original-data/UAV_Gazebo_data/UAV_takeoff_circle.txt" # for drone data

# Constant for noise filter 
MOVING_WINDOW_SIZE = 10 # Moving average of 10 data points 
SIGMA = 0.3 # Standard deviation for Gaussian filter

# Noise level to add to the data (use this for generated data with no noise)
NOISE_LEVEL = 0.01

# # Gazebo drone data 
# # STLSQ constants
# THRESHOLD_MIN = 0.00
# THRESHOLD_MAX  = 0.1
# NUMBER_OF_THRESHOLD_VALUES = 11
# MAX_ITERATIONS = 10

# # Data generation constants
# PREDICTION_TIME = 100 # time to predict (in milliseconds)
# WINDOW_SIZE = 500 # data taken into account for prediction
# PREDICTION_FREQUENCY = 500 # frequency of prediction (in milliseconds)

# Projectile motion problem
# STLSQ constants 
THRESHOLD_MIN = 0.00
THRESHOLD_MAX  = 0.2
NUMBER_OF_THRESHOLD_VALUES = 11
MAX_ITERATIONS = 10

# Data generation constants
dt = 0.05 # Time step (s)
TIME_OF_DATA = 15
PREDICTION_TIME = 0.5 # time to predict (in seconds)
WINDOW_SIZE = 3 # data taken into account for prediction
PREDICTION_FREQUENCY = 3

# starting point of ball 
x0 = 30
y0 = 40
z0 = 60

# Initial conditions
v0 = 40  # Initial velocity (m/s)
launch_angle = 30  # Launch angle in degrees

# Real drone data 
# # Data generation constants
# TIME_OF_DATA = 2
# PREDICTION_TIME = 0.05 # time to predict (in seconds)
# WINDOW_SIZE = 0.2 # data taken into account for prediction
# PREDICTION_FREQUENCY = 0.5 # frequency of prediction (in seconds)




