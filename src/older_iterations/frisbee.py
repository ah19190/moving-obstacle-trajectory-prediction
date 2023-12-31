import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from frispy.disc import Disc

plt.rc("font", size=14, family="serif")
# Negative theta is an airbounce
disc2 = Disc(vx=40, vy = 30, vz = 15, theta=-0.15, phi=0.5)
result2, _ = disc2.compute_trajectory(flight_time=10)

phi = result2["phi"][50]
theta = result2["theta"][50]
r = disc2.eom.rotation_matrix_from_phi_theta(phi, theta)

from matplotlib.tri import Triangulation
def get_edge(radii, n=20):
    alpha = np.linspace(0, 2*np.pi, n)
    r = radii * np.ones(n)
    x = r * np.cos(alpha)
    y = r * np.sin(alpha)
    z = np.zeros(n)
    return np.array([x, y, z]).T

edge = get_edge(disc2.eom.diameter / 2 * 10)
edge_T = edge @ r
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot3D(result2["x"], result2["y"], result2["z"], "black", label='Coordinate data')

# plot the start point 
ax.scatter(result2["x"][0], result2["y"][0], result2["z"][0], c="red", marker="o", s=100, label='Start point')
# Set labels and title
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Frisbee Data') 

ax.plot_trisurf(edge_T[:, 0] + result2["x"][30], edge_T[:, 1] + result2["y"][30], edge_T[:, 2] + result2["z"][30])
ax.set_xlim(0, 40)
ax.set_ylim(-20, 20)
ax.set_zlim(0, 10)

fig.set_size_inches(8, 8)
plt.legend()
plt.show()

selected_features = ['times', 'x', 'y', 'z']

# Create a new dictionary containing only the selected features
selected_data = {feature: result2[feature] for feature in selected_features}

# Create a DataFrame from the selected data
df = pd.DataFrame(selected_data)

# Define the CSV file path where you want to save the data
csv_file_path = 'selected_data.csv'

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)