"Constants and common code."

from pathlib import Path

# Directory names.
ORIGINAL_DATA_DIR = str(Path(Path(__file__).parent.parent, "original-data"))
DATA_DIR = str(Path(Path(__file__).parent.parent, "data"))
OUTPUT_DIR = str(Path(Path(__file__).parent.parent, "output"))

# STLSQ constants 
THRESHOLD_MIN = 0.01
THRESHOLD_MAX  = 0.1
NUMBER_OF_THRESHOLD_VALUES = 11
MAX_ITERATIONS = 10

# Noise level for the data
NOISE_LEVEL = 0.02

# Constants for the projectile motion problem
dt = 0.1 # Time step (s)
TIME_OF_DATA = 6
PREDICTION_TIME = 1

# starting point of ball 
x0 = 40
y0 = 20
z0 = 10

# Initial conditions
v0 = 30  # Initial velocity (m/s)
launch_angle = 30  # Launch angle in degrees