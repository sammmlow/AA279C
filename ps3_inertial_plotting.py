
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom code
import source.attitudes as att

# Define the default osculating orbital elements [km and degrees].
initial_elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6]
initial_omega = [0.0873, 0.0698, 0.0524]                   # rad/s
initial_inertia = np.diag([4770.398, 6313.894, 7413.202])  # kg m^2

zero_torque = np.array([0.0, 0.0, 0.0]) # N m

initial_angular_momentum = np.dot(initial_inertia, initial_omega)
print(f"Initial angular momentum: {initial_angular_momentum} kg m^2/s")


# Load in the CSV data from each satellite as pandas dataframes
mrp_satellite_df = pd.read_csv("ps3_data/mrp_satellite.csv")
print(mrp_satellite_df)
qtr_satellite_df = pd.read_csv("ps3_data/qtr_satellite.csv")
print(qtr_satellite_df)


if False:
    # Plot the angular velocities in the principal frame, to check
    plt.figure()

    for sat_name, sat_df in zip(["MRP", "QTR"], [mrp_satellite_df, qtr_satellite_df]):

        for direction in ["x", "y", "z"]:
            plt.plot(sat_df["time"], sat_df[f"w{direction}"], 
                    label=f"{sat_name} {direction}",
                    linestyle="--" if sat_name == "MRP" else ":")
        
    plt.xlabel("Time [s]")
    plt.ylabel("Angular Velocity [rad/s]")
    plt.legend()
    plt.grid()
    plt.show()

if False:
    # Plot parametrically as the polhode in 3D
    plt.figure()
    ax = plt.axes(projection='3d')

    for sat_name, sat_df in zip(["MRP", "QTR"], [mrp_satellite_df, qtr_satellite_df]):
        ax.plot3D(sat_df["wx"], sat_df["wy"], sat_df["wz"], 
                label=sat_name,
                linestyle="--" if sat_name == "MRP" else ":")
        
    ax.set_xlabel("wx [rad/s]")
    ax.set_ylabel("wy [rad/s]")
    ax.set_zlabel("wz [rad/s]")

    ax.axis("equal")

    ax.legend()
    plt.show()

# The attitudes are the conversion from the principal frame to the inertial frame
# We will plot the xyz triad as RGB lines using the DCM rotation matrix in 3D.
# Only plot one satellite at a time

num_skip = 5
max_i = 2600
num_times_plotted = max_i // num_skip
alpha_linear_scaled = np.linspace(0, 1, num_times_plotted, endpoint=True)
print(f"Length of alpha scaling is {len(alpha_linear_scaled)}")

# Viewpoint in 3D for the plot
viewpoint = {"elev": 20, "azim": -10}

for sat_name, sat_df in zip(["MRP", "QTR"], [mrp_satellite_df, qtr_satellite_df]):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Set the viewpoint
    ax.view_init(elev=viewpoint["elev"], azim=viewpoint["azim"])

    # Make an attitude class to convert the MRP or QTR to a DCM
    if sat_name == "MRP":
        sat_att = att.MRP()
    else:
        sat_att = att.QTR()
    
    for i, row in sat_df.iterrows():

        if i % num_skip != 0:
            continue

        if i >= max_i:
            break

        # Convert the MRP or QTR to a DCM
        if sat_name == "MRP":
            dcm = sat_att._mrp2dcm([row["a0"], row["a1"], row["a2"]])
        else:
            dcm = sat_att._qtr2dcm([row["a0"], row["a1"], row["a2"], row["a3"]])

        x, y, z = dcm
        # print(f"accessing index {i // num_skip} with i = {i}")
        alpha = alpha_linear_scaled[i // num_skip]
        ax.plot3D([0, x[0]], [0, x[1]], [0, x[2]], "r", alpha=alpha)
        ax.plot3D([0, y[0]], [0, y[1]], [0, y[2]], "g", alpha=alpha)
        ax.plot3D([0, z[0]], [0, z[1]], [0, z[2]], "b", alpha=alpha)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    ax.axis("equal")
    
    # Use more human names for the title
    if sat_name == "MRP":
        title_name = "Modified Rodrigues Parameters"
    else:
        title_name = "Quaternions"

    plt.title(title_name)
    plt.show()

    # Save the figure
    fig.savefig(f"ps3_data/{sat_name}_attitude_inertial_max{max_i}_skip{num_skip}.png")



# Convert the angular velocities to the body frame, and plot them
# This is the herpolhode plot



# Calculate the angular momentum in the body frame, and plot it
# It should be constant, and equal to the initial angular momentum


