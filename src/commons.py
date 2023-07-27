"Constants and common code."

# Directory names.
ORIGINAL_DATA_DIR = str(Path(Path(__file__).parent.parent, "original-data"))
DATA_DIR = str(Path(Path(__file__).parent.parent, "data"))
OUTPUT_DIR = str(Path(Path(__file__).parent.parent, "output"))

# STLSQ constants 
THRESHOLD = 0.09
MAX_ITERATIONS = 10

# Noise level for the data
NOISE_LEVEL = 0.02