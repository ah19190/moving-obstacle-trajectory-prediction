"""Utilities related to graphing."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# define the graphing functions
def style_axis2d(ax, xlabel: str, ylabel: str):
    """Styles a 2D graph.
    """
    very_light_gray = "#efefef"
    light_gray = "#999999"
    dark_gray = "#444444"

    ax.set_xlabel(xlabel, {"color": dark_gray})
    ax.set_ylabel(ylabel, {"color": dark_gray})
    ax.set_title(ax.get_title(), {"color": dark_gray})
    ax.tick_params(axis="x", colors=light_gray)
    ax.tick_params(axis="y", colors=light_gray)
    ax.set_facecolor(very_light_gray)
    for spine in ax.spines.values():
        spine.set_edgecolor(light_gray)


def graph_result(u: np.ndarray, u_approximation_x: np.ndarray,
                 u_approximation_y: np.ndarray,u_approximation_z: np.ndarray,
                 t: np.ndarray) -> None:
    """Graphs x(t), y(t) and z(t) of the original trajectory and the SINDy
    """
    orange = "#EF6C00"
    teal = "#007b96"

    fig = plt.figure(figsize=plt.figaspect(1))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1, sharey=ax1)
    fig.tight_layout(pad=3)

    ax1.plot(t[:], u[:, 0], color=orange, linewidth=0.8)
    ax1.plot(t[:], u_approximation_x[:, 0], color=teal, linewidth=0.4)
    ax1.set_title("x(t)")
    style_axis2d(ax1, "t", "x")
    ax1.legend(labels=["Original trajectory", "SINDy approximation"],
               fontsize=8)

    ax2.plot(t[:], u[:, 1], color=orange, linewidth=0.8)
    ax2.plot(t[:], u_approximation_y[:, 0], color=teal, linewidth=0.4)
    ax2.set_title("y(t)")
    style_axis2d(ax2, "t", "y")
    ax2.legend(labels=["Original trajectory", "SINDy approximation"],
               fontsize=8)
    
    ax3.plot(t[:], u[:, 2], color=orange, linewidth=0.8)
    ax3.plot(t[:], u_approximation_z[:, 0], color=teal, linewidth=0.4)
    ax3.set_title("z(t)")
    style_axis2d(ax3, "t", "z")
    ax3.legend(labels=["Original trajectory", "SINDy approximation"],
               fontsize=8)

    plt.show()

    def 3d_graph_result(u: np.ndarray, u_approximation_x: np.ndarray,
                 u_approximation_y: np.ndarray,u_approximation_z: np.ndarray,
                 t: np.ndarray) -> None:
    """Graphs the original trajectory and the SINDy trajecctory in 3D space
    """

    # Model the trajectory in 3D 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Projectile Motion')  
    
    # Plot the starting point 
    ax.scatter(u[0, 0], u[0, 1], u[0, 2], color='red', label='Start')

    # Plot the trajectory of the obstacle (without noise)
    ax.plot(u[:, 0], u[:, 1], u[: , 2], color='black', label='Ground truth')

    # Plot the SINDy approximation of the trajectory
    ax.plot(u_approximation_x, u_approximation_y, u_approximation_z, color='blue', label='SINDy')

    plt.legend()
    plt.show()


