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

