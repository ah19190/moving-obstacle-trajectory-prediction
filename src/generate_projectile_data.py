"Generate the data for a projectile motion problem."

# Constants 
g = 9.81  # Acceleration due to gravity (m/s^2)

# starting point of ball )
x0 = 40
y0 = 50
z0 = 10

# Initial conditions
v0 = 40  # Initial velocity (m/s)
launch_angle = 30  # Launch angle in degrees

# Time points for the trajectory
t = np.arange(0, 2*v0/g, dt)

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

