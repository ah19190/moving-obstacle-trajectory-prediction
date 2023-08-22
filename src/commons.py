"Constants and common code."
from pathlib import Path

# Directory names.
ORIGINAL_DATA_DIR = str(Path(Path(__file__).parent.parent, "original-data"))
DATA_DIR = str(Path(Path(__file__).parent.parent, "data"))
OUTPUT_DIR = str(Path(Path(__file__).parent.parent, "output"))

# Data file we want to parse for trajectory data 
TRAJECTORY_DATA_FILE = "../original-data/drone_trajectory_data.csv"

# STLSQ constants 
THRESHOLD_MIN = 0.0
THRESHOLD_MAX  = 0.0
NUMBER_OF_THRESHOLD_VALUES = 2
MAX_ITERATIONS = 10

# Constants for the projectile motion problem
dt = 0.01 # Time step (s)
TIME_OF_DATA = 2
PREDICTION_TIME = 0.5 # time to predict (in seconds)
WINDOW_SIZE = 2 # data taken into account for prediction

# Constants for the drone problem
PREDICTION_FREQUENCY = 0.5 # frequency of prediction (in seconds)

# Noise level to add to the data (use this for generated data with no noise)
NOISE_LEVEL = 0.00

# starting point of ball 
x0 = 30
y0 = 40
z0 = 60

# Initial conditions
v0 = 40  # Initial velocity (m/s)
launch_angle = 30  # Launch angle in degrees


