"""Utilities related to graphing."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# define the graphing functions
def style_axis2d(ax, xlabel: str, ylabel: str):
    """Styles a 2D graph. Used in graph_result and graph_result_prediction_only.
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

def graph_result(u: np.ndarray, u_approximation: np.ndarray,
                 t: np.ndarray, t_predict: np.ndarray) -> None:
    """Graphs x(t), y(t) and z(t) of the original trajectory and the SINDy 
    including the window of data used for fitting
    """
    # Extract X, Y, and Z coordinates from the data
    u_approximation_x = u_approximation[:, 0:1]
    u_approximation_y = u_approximation[:, 1:2]
    u_approximation_z = u_approximation[:, 2:3]

    orange = "#EF6C00"
    teal = "#007b96"

    fig = plt.figure(figsize=plt.figaspect(1))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1, sharey=ax1)
    fig.tight_layout(pad=3)

    ax1.plot(t[:], u[:, 0], color=orange, linewidth=0.8)
    ax1.plot(t_predict[:], u_approximation_x[:, 0], color=teal, linewidth=0.4)
    ax1.set_title("x(t)")
    style_axis2d(ax1, "t", "x")
    ax1.legend(labels=["Original trajectory", "SINDy approximation"],
               fontsize=8)

    ax2.plot(t[:], u[:, 1], color=orange, linewidth=0.8)
    ax2.plot(t_predict[:], u_approximation_y[:, 0], color=teal, linewidth=0.4)
    ax2.set_title("y(t)")
    style_axis2d(ax2, "t", "y")
    ax2.legend(labels=["Original trajectory", "SINDy approximation"],
               fontsize=8)
    
    ax3.plot(t[:], u[:, 2], color=orange, linewidth=0.8)
    ax3.plot(t_predict[:], u_approximation_z[:, 0], color=teal, linewidth=0.4)
    ax3.set_title("z(t)")
    style_axis2d(ax3, "t", "z")
    ax3.legend(labels=["Original trajectory", "SINDy approximation"],
               fontsize=8)

    plt.show()

def graph_result_prediction_only(u: np.ndarray, u_approximation: np.ndarray,
                t_predict: np.ndarray) -> None:
    """Graphs x(t), y(t) and z(t) of the original trajectory and the SINDy
    excluding the window of data used for fitting 
    """
    # Extract X, Y, and Z coordinates from the data
    u_approximation_x = u_approximation[:, 0:1]
    u_approximation_y = u_approximation[:, 1:2]
    u_approximation_z = u_approximation[:, 2:3]

    orange = "#EF6C00"
    teal = "#007b96"

    fig = plt.figure(figsize=plt.figaspect(1))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1, sharey=ax1)
    fig.tight_layout(pad=3)

    ax1.plot(t_predict[:], u[:, 0], color=orange, linewidth=0.8)
    ax1.plot(t_predict[:], u_approximation_x[:, 0], color=teal, linewidth=0.4)
    ax1.set_title("x(t)")
    style_axis2d(ax1, "t", "x")
    ax1.legend(labels=["Original trajectory", "SINDy approximation"],
               fontsize=8)

    ax2.plot(t_predict[:], u[:, 1], color=orange, linewidth=0.8)
    ax2.plot(t_predict[:], u_approximation_y[:, 0], color=teal, linewidth=0.4)
    ax2.set_title("y(t)")
    style_axis2d(ax2, "t", "y")
    ax2.legend(labels=["Original trajectory", "SINDy approximation"],
               fontsize=8)
    
    ax3.plot(t_predict[:], u[:, 2], color=orange, linewidth=0.8)
    ax3.plot(t_predict[:], u_approximation_z[:, 0], color=teal, linewidth=0.4)
    ax3.set_title("z(t)")
    style_axis2d(ax3, "t", "z")
    ax3.legend(labels=["Original trajectory", "SINDy approximation"],
               fontsize=8)

    plt.show()

def graph_error(u: np.ndarray, u_approximation: np.ndarray,
                t_predict: np.ndarray) -> None:
    """Graphs the error in x(t), y(t) and z(t) of the original trajectory and the SINDy
    excluding the window of data used for fitting 
    """
    # Get the modulo difference between u and u_approximation 
    u_diff = np.abs(u - u_approximation)

    # Extract X, Y, and Z coordinates from the data
    u_diff_x = u_diff[:, 0:1]
    u_diff_y = u_diff[:, 1:2]
    u_diff_z = u_diff[:, 2:3]
    
    fig = plt.figure(figsize=plt.figaspect(1))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1, sharey=ax1)
    fig.tight_layout(pad=3)

    ax1.plot(t_predict[:], u_diff[:, 0], linewidth=1)
    ax1.set_title("x(t)")
    style_axis2d(ax1, "t", "x")
    ax1.legend(labels=["Error"],
               fontsize=8)

    ax2.plot(t_predict[:], u_diff[:, 1], linewidth=1)
    ax2.set_title("y(t)")
    style_axis2d(ax2, "t", "y")
    ax2.legend(labels=["Error"],
               fontsize=8)
    
    ax3.plot(t_predict[:], u_diff[:, 2], linewidth=1)
    ax3.set_title("z(t)")
    style_axis2d(ax3, "t", "z")
    ax3.legend(labels=["Error"],
               fontsize=8)

def three_d_graph_result(u: np.ndarray, u_window: np.ndarray, u_approximation: np.ndarray) -> None:
    """Graphs the original trajectory and the SINDy trajectory in 3D space
    """
    # Extract X, Y, and Z coordinates from the data
    u_approximation_x = u_approximation[:, 0:1]
    u_approximation_y = u_approximation[:, 1:2]
    u_approximation_z = u_approximation[:, 2:3]
    
    # Model the trajectory in 3D 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    # ax.set_title('3D Projectile Motion')
    ax.set_title('Gazebo Drone Data')
    # ax.set_title('Frisbee Data')    

    # If you want to plot against the whole data set (not just the window)
    # Plot the starting point of the data
    ax.scatter(u[0, 0], u[0, 1], u[0, 2], color='red', label='Start of coordinate data')
    # Plot the trajectory of the obstacle (without noise)
    ax.plot(u[:, 0], u[:, 1], u[: , 2], color='black', label='Ground truth')

    # # Plot just the window 
    # # Plot the trajectory of the obstacle of the window
    # ax.plot(u_window[:, 0], u_window[:, 1], u_window[: , 2], color='black', label='Window')

    # Plot the starting point of the window
    ax.scatter(u_window[0, 0], u_window[0, 1], u_window[0, 2], color='blue', label='Start of window')

    # Plot the starting point of the SINDy approximation
    ax.scatter(u_approximation_x[0, 0], u_approximation_y[0, 0], u_approximation_z[0, 0], color='green', label='Start of SINDy approximation')

    # Plot the SINDy approximation of the trajectory
    ax.plot(u_approximation_x, u_approximation_y, u_approximation_z, color='blue', label='SINDy')

    plt.legend()
    plt.show()

def three_d_graph_result_ensemble(coordinate_data_fit: np.ndarray, coordinate_ground_truth: np.ndarray,
                t_predict: np.ndarray,  ensemble_coefs, model_all) -> None:
    """Graphs the original trajectory and the SINDy trajectory in 3D space with all the ensemble models
    """
    simulations = []
    # Simulate the trajectory using the ensemble models and add it to a list
    for coefs in ensemble_coefs:
        model_all.coef = coefs
        simulate_data = model_all.simulate(coordinate_data_fit[-1, :], t_predict, integrator="odeint")
        simulations.append(simulate_data)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the trajectory of the obstacle of the window
    ax.plot(coordinate_ground_truth[:, 0], coordinate_ground_truth[:, 1], coordinate_ground_truth[: , 2], color='black', label='Window')

    # Plot the starting point of the window
    ax.scatter(coordinate_ground_truth[0, 0], coordinate_ground_truth[0, 1], coordinate_ground_truth[0, 2], color='blue', label='Start of window')

    # Plot the starting point of the SINDy approximation
    ax.scatter(coordinate_data_fit[-1, 0], coordinate_data_fit[-1, 1], coordinate_data_fit[-1, 2], color='green', label='Start of SINDy approximation')
    
    for i, simulate_data in enumerate(simulations):
        # Extract X, Y, and Z coordinates from the simulation data
        simulate_data_x = simulate_data[:, 0]
        simulate_data_y = simulate_data[:, 1]
        simulate_data_z = simulate_data[:, 2]
        
        # Plot the trajectory of the simulation
        ax.plot(simulate_data_x, simulate_data_y, simulate_data_z, color='red')   

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Frisbee Data')       

    plt.legend()
    plt.show()

def three_d_graph_result_ground_vs_noisy(u_ground_truth: np.ndarray,u_noisy: np.ndarray, t: np.ndarray) -> None:
    """Graphs the ground truth vs the noisy data in 3D space (for data which we add noise using rmse)
    Use this to check the how noise affects the ground truth data 
    """
    # Model the trajectory in 3D 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    # ax.set_title('3D Projectile Motion')
    ax.set_title('Gazebo Drone Data')  

    # Plot the starting point of the data
    ax.scatter(u_ground_truth[0, 0], u_ground_truth[0, 1], u_ground_truth[0, 2], color='red', label='Start of training data')

    # Plot the trajectory of the obstacle
    ax.plot(u_ground_truth[:, 0], u_ground_truth[:, 1], u_ground_truth[: , 2], color='black', label='Ground truth')

    # Plot the trajectory of the obstacle (cleaned)
    ax.plot(u_noisy[:, 0], u_noisy[:, 1], u_noisy[: , 2], color='red', label='Cleaned')

    plt.legend()
    plt.show()