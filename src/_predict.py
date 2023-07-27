"""Predicts a trajectory using the SINDy model."""

from utils_graph import graph_result, three_d_graph_result

def main() -> None:
    # logging.info("Predicting.")
    
    # Need to generate the data and call the fit function here

    # Prediction part 
    u0x = np.array([u[0, 0], xdot[0, 0]])
    u_approximation_x = modelx.simulate(u0x, t)

    u0y = np.array([u[0, 1], ydot[0, 0]])
    u_approximation_y = modely.simulate(u0y, t)

    u0z = np.array([u[0, 2], zdot[0, 0]])
    u_approximation_z = modelz.simulate(u0z, t)

    graph_result(u, u_approximation_x, u_approximation_y, u_approximation_z, t)

    three_d_graph_result(u, u_approximation_x, u_approximation_y, u_approximation_z, t)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
