"Constants and common code."
from pathlib import Path

# Directory names
ORIGINAL_DATA_DIR = str(Path(Path(__file__).parent.parent, "original-data"))
DATA_DIR = str(Path(Path(__file__).parent.parent, "data"))
OUTPUT_DIR = str(Path(Path(__file__).parent.parent, "output"))

# Constant for noise filter 
MOVING_WINDOW_SIZE = 15 # Moving average of 10 data points 
SIGMA = 0.3 # Standard deviation for Gaussian filter

# Noise level to add to the data (use this for generated data with no noise)
NOISE_LEVEL = 0.00

# Projectile motion problem
# STLSQ constants 
THRESHOLD_MIN = 0.00
THRESHOLD_MAX  = 0.3
NUMBER_OF_THRESHOLD_VALUES = 21
MAX_ITERATIONS = 10

# Data generation constants
dt = 0.05 # Time step (s)
TIME_OF_DATA = 15
PREDICTION_TIME = 0.5 # time to predict (in seconds)
WINDOW_SIZE = 3 # data taken into account for prediction
PREDICTION_FREQUENCY = 3

# new constant for adaptive window size
MIN_WINDOW_SIZE = 1
MAX_WINDOW_SIZE = 4

# starting point of ball 
x0 = 30
y0 = 40
z0 = 60

# Initial conditions
v0 = 40  # Initial velocity (m/s)
launch_angle = 30  # Launch angle in degrees

