"Generate the data for a projectile motion problem."

from pathlib import Path

import h5py
import numpy as np 

# Constants 
g = 9.81  # Acceleration due to gravity (m/s^2)

dt = 0.1  # Time step (s)

# starting point of ball )
x0 = 40
y0 = 50
z0 = 10

# Initial conditions
v0 = 40  # Initial velocity (m/s)
launch_angle = 30  # Launch angle in degrees

# Function to calculate the projectile motion (remove z to follow the example)
def projectile_motion(v0, theta_deg, t):
    theta_rad = np.radians(theta_deg)
    vx0 = v0 * np.cos(theta_rad)
    vy0 = v0 * np.sin(theta_rad)
    vz0 = v0 * np.cos(theta_rad)

    x = x0 + vx0 * t
    y = y0 + vy0 * t - 0.5 * g * t**2
    z = z0 + vz0 * t - 0.5 * g * t**2

    return np.array([x, y, z])
    # return np.array([x, y])

def main() -> None:
    # logging.info("Generating data.")

    # Time points for the trajectory
    t = np.arange(0, 2*v0/g, dt)

    # Calculate the projectile motion
    u = projectile_motion(v0, launch_angle, t).T

    data_file_path = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_path, "w") as file:
        file.create_dataset(name="u", data=u)
        file.create_dataset(name="t", data=t)

if __name__ == "__main__":
    main()