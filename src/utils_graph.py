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


def three_d_graph_result(u: np.ndarray, u_approximation_x: np.ndarray,
                u_approximation_y: np.ndarray,u_approximation_z: np.ndarray,
                t: np.ndarray) -> None:
    """Graphs the original trajectory and the SINDy trajectory in 3D space
    """
    # Model the trajectory in 3D 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Projectile Motion')  

    # Plot the starting point of the data
    ax.scatter(u[0, 0], u[0, 1], u[0, 2], color='red', label='Start of training data')

    # Plot the trajectory of the obstacle (without noise)
    ax.plot(u[:, 0], u[:, 1], u[: , 2], color='black', label='Ground truth')

    # Plot the starting point of the SINDy approximation
    ax.scatter(u_approximation_x[0, 0], u_approximation_y[0, 0], u_approximation_z[0, 0], color='green', label='Start of SINDy approximation')

    # Plot the SINDy approximation of the trajectory
    ax.plot(u_approximation_x, u_approximation_y, u_approximation_z, color='blue', label='SINDy')

    plt.legend()
    plt.show()

def three_d_graph_result_new(u: np.ndarray,u2: np.ndarray, t: np.ndarray) -> None:
    """Graphs the original trajectory and the SINDy trajectory in 3D space
    """
    # Model the trajectory in 3D 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Projectile Motion')  

    # Plot the starting point of the data
    ax.scatter(u[0, 0], u[0, 1], u[0, 2], color='red', label='Start of training data')

    # Plot the trajectory of the obstacle (with noise)
    ax.plot(u[:, 0], u[:, 1], u[: , 2], color='black', label='Noisy data')

    # Plot the trajectory of the obstacle (cleaned)
    ax.plot(u2[:, 0], u2[:, 1], u2[: , 2], color='red', label='Cleaned')

    plt.legend()
    plt.show()

def fft_denoiser(x, n_components, to_real=True):
    """Fast fourier transform denoiser.
    
    Denoises data using the fast fourier transform.
    
    Parameters
    ----------
    x : numpy.array
        The data to denoise.
    n_components : int
        The value above which the coefficients will be kept.
    to_real : bool, optional, default: True
        Whether to remove the complex part (True) or not (False)
        
    Returns
    -------
    clean_data : numpy.array
        The denoised data.
        
    References
    ----------
    .. [1] Steve Brunton - Denoising Data with FFT[Python]
       https://www.youtube.com/watch?v=s2K1JfNR7Sc&ab_channel=SteveBrunton
    
    """
    n = len(x)
    
    # compute the fft
    fft = np.fft.fft(x, n)
    
    # compute power spectrum density
    # squared magnitud of each fft coefficient
    PSD = fft * np.conj(fft) / n

    # Plot the power spectrum density
    # plt.plot(PSD)
    # plt.show()

    # keep high frequencies
    _mask = PSD > n_components
    fft = _mask * fft
    
    # inverse fourier transform
    clean_data = np.fft.ifft(fft)
    
    if to_real:
        clean_data = clean_data.real
    
    return clean_data

def moving_average_filter(data, window_size):
    """Moving average filter.
    """
    window = np.ones(window_size) / window_size
    smoothed_data = np.apply_along_axis(lambda x: np.convolve(x, window, mode='same'), axis=0, arr=data)
    
    for i in range(window_size // 2):
        smoothed_data[i] = data[i]
        smoothed_data[-i - 1] = data[-i - 1]
    
    return smoothed_data

