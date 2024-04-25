
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

initial_angular_momentum = initial_inertia @ initial_omega
print(f"Initial angular momentum: {initial_angular_momentum} kg m^2/s")


# Load in the CSV data from each satellite as pandas dataframes
mrp_satellite_df = pd.read_csv("ps3_data/mrp_satellite.csv")
print(mrp_satellite_df)
qtr_satellite_df = pd.read_csv("ps3_data/qtr_satellite.csv")
print(qtr_satellite_df)


if True:
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

if True:
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

num_skip = 2000
len_tot = min(len(mrp_satellite_df), len(qtr_satellite_df))
max_i = min(len_tot, np.inf)
num_times_plotted = max_i // num_skip + 1
alpha_linear_scaled = np.linspace(0, 1, num_times_plotted, endpoint=True)
print(f"Length of alpha scaling is {len(alpha_linear_scaled)}")
# Viewpoint in 3D for the plot
viewpoint = {"elev": 20, "azim": -10}

if True:

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

            # Check that each is a unit vector
            assert np.isclose(np.linalg.norm(x), 1.0)
            assert np.isclose(np.linalg.norm(y), 1.0)
            assert np.isclose(np.linalg.norm(z), 1.0)

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

# First, calculate the angular velocities in the inertial frame
omega_inertial_history_per_sat = []
dcms_history_per_sat = []
times_history_per_sat = []

for sat_name, sat_df in zip(["MRP", "QTR"], [mrp_satellite_df, qtr_satellite_df]):

    # Make list to store the history of the angular velocities
    omega_inertial_history = []
    dcm_history = []
    times_history = []

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

        # Store the time
        times_history.append(row["time"])

        # Convert the MRP or QTR to a DCM
        if sat_name == "MRP":
            dcm = sat_att._mrp2dcm([row["a0"], row["a1"], row["a2"]])
        else:
            dcm = sat_att._qtr2dcm([row["a0"], row["a1"], row["a2"], row["a3"]])

        # Rotate the angular velocity to the body frame
        omega_body = np.array([row["wx"], row["wy"], row["wz"]])
        omega_inertial = dcm.T @ omega_body
        # omega_inertial = np.dot(dcm, omega_body)

        # Store the angular velocity in the inertial frame
        omega_inertial_history.append(omega_inertial)
        # Store the DCM
        dcm_history.append(dcm)

        # Store the times for each satellite
        times_history_per_sat.append(times_history)
    
    # Store the history of the angular velocities and the DCMs
    omega_inertial_history_per_sat.append(omega_inertial_history)
    dcms_history_per_sat.append(dcm_history)


# Then we will plot the angular velocities in the inertial frame

# Do this per satellite
if True:
    for sat_name, sat_ang_vel in zip(["MRP", "QTR"], omega_inertial_history_per_sat):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # Set the viewpoint
        ax.view_init(elev=viewpoint["elev"], azim=viewpoint["azim"])

        # Stack the history of the angular velocities as np array (time, 3)
        omega_inertial_history = np.array(sat_ang_vel)
        assert omega_inertial_history.shape[1] == 3

        # Just plot the tip of the angular velocity vector
        # Plot by alpha which requires plotting each point separately
        for (wxi, wyi, wzi), alpha in zip(sat_ang_vel, alpha_linear_scaled):
            ax.scatter3D(wxi, wyi, wzi, color="k", alpha=alpha)

        # ax.scatter3D(omega_inertial_history[:, 0], 
        #              omega_inertial_history[:, 1], 
        #              omega_inertial_history[:, 2], 
        #              c=alpha_linear_scaled,
        #              cmap="gray",
        #              alpha=alpha_linear_scaled)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        ax.axis("equal")
        
        # Use more human names for the title
        if sat_name == "MRP":
            title_name = "Herpolhode when using the Modified Rodrigues Parameters"
        else:
            title_name = "Herpolhode when using the Quaternions"

        plt.title(title_name)
        plt.show()

        # Save the figure
        fig.savefig(f"ps3_data/{sat_name}_angular_velocity_body_inertial_max{max_i}_skip{num_skip}.png")


# Calculate the angular momentum in the body frame, and plot it
# It should be constant, and equal to the initial angular momentum

# Do this per satellite. Again calculate first (no plotting).
angular_momentum_history_per_sat = []

for sat_name, sat_ang_vel, sat_dcm in zip(["MRP", "QTR"], 
                                          omega_inertial_history_per_sat,
                                          dcms_history_per_sat):

    # Make list to store the history of the angular velocities
    angular_momentum_history = []

    for omega_inertial_curr, dcm_curr in zip(sat_ang_vel, sat_dcm):
        # Calculate the angular momentum in the inertial frame
        # L = I @ omega
        # Which means we need I in the inertial frame
        inertia_inertial_frame = dcm_curr.T @ initial_inertia @ dcm_curr
        # inertia_inertial_frame = dcm_curr @ initial_inertia @ dcm_curr.T

        angular_momentum_inertial = inertia_inertial_frame @ omega_inertial_curr
        angular_momentum_history.append(angular_momentum_inertial)

        # print(np.dot(angular_momentum_inertial, omega_inertial_curr))
    
    # Store the history of the angular velocities
    angular_momentum_history_per_sat.append(angular_momentum_history)

# Now plot the angular momentum in the inertial frame per satellite

if True:
    for sat_name, sat_ang_mom in zip(["MRP", "QTR"], angular_momentum_history_per_sat):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # Set the viewpoint
        ax.view_init(elev=viewpoint["elev"], azim=viewpoint["azim"])

        # Stack the history of the angular velocities as np array (time, 3)
        angular_momentum_history = np.array(sat_ang_mom)
        assert angular_momentum_history.shape[1] == 3

        # Just plot the tip of the angular momentum vector
        # Plot by alpha which requires plotting each point separately
        for (lxi, lyi, lzi), alpha in zip(sat_ang_mom, alpha_linear_scaled):
            ax.scatter3D(lxi, lyi, lzi, color="k", alpha=alpha)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        ax.axis("equal")
        
        # Use more human names for the title
        if sat_name == "MRP":
            title_name = "Angular Momentum when using the Modified Rodrigues Parameters"
        else:
            title_name = "Angular Momentum when using the Quaternions"

        plt.title(title_name)
        plt.show()

        # Save the figure
        fig.savefig(f"ps3_data/{sat_name}_angular_momentum_body_inertial_max{max_i}_skip{num_skip}.png")


# Check the norm of the angular momentum, it should be constant

if True:
    fig = plt.figure()

    print("\nPlotting norm of the angular momentum\n")

    for sat_name, sat_ang_mom, sat_times in zip(["MRP", "QTR"], 
                                                angular_momentum_history_per_sat,
                                                times_history_per_sat):
        print(f"Length of angular momentum history for {sat_name} is {len(sat_ang_mom)}")
        angular_momentum_norm = np.linalg.norm(np.array(sat_ang_mom), axis=1)

        # Plot the norm of the angular momentum    
        plt.plot(sat_times, angular_momentum_norm, label=sat_name,
                linestyle="--" if sat_name == "MRP" else ":")
        
        # Plot the x component of the angular momentum to sanity check
        # plt.plot(sat_times, np.array(sat_ang_mom)[:, 0], label=f"{sat_name} x",
        #          linestyle="--" if sat_name == "MRP" else ":")

    print("Finished for loop\n")

    plt.xlabel("Time [s]")
    plt.ylabel("Angular Momentum Norm [kg m^2/s]")
    plt.legend()
    plt.grid()
    plt.show()

    # Save the figure
    fig.savefig(f"ps3_data/angular_momentum_norm_max{max_i}_skip{num_skip}.png")


# Plot four subplots with the angular momentum components and norm for each satellite
fig, axs = plt.subplots(2, 2)

for sat_name, sat_ang_mom, sat_times in zip(["MRP", "QTR"],
                                            angular_momentum_history_per_sat,
                                            times_history_per_sat):
    linestyle_curr = "--" if sat_name == "MRP" else ":"

    axs[0, 0].plot(sat_times, np.array(sat_ang_mom)[:, 0], label=f"{sat_name} x",
                linestyle=linestyle_curr)
    axs[0, 1].plot(sat_times, np.array(sat_ang_mom)[:, 1], label=f"{sat_name} y",
                linestyle=linestyle_curr)
    axs[1, 0].plot(sat_times, np.array(sat_ang_mom)[:, 2], label=f"{sat_name} z",
                linestyle=linestyle_curr)
    axs[1, 1].plot(sat_times, np.linalg.norm(np.array(sat_ang_mom), axis=1), label=f"{sat_name} norm",
                linestyle=linestyle_curr)

# Set the labels in a loop
for i, (ax, ax_param) in enumerate(zip(axs.flat, ["x", "y", "z", "norm"])):
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"L {ax_param} [kg m^2/s]")
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()

# Save the figure
fig.savefig(f"ps3_data/angular_momentum_components_max{max_i}_skip{num_skip}.png")

# Now that we have the angular momentum in the inertial frame, we can calculate the
# plane of the herpolhode, and plot it with the angular velocity in the inertial frame

# Do this per satellite.

fig = plt.figure()
for sat_name, sat_ang_vel, sat_ang_mom, sat_times in zip(["MRP", "QTR"], 
                                              omega_inertial_history_per_sat,
                                              angular_momentum_history_per_sat,
                                              times_history_per_sat):

    # Stack the history of the angular velocities as np array (time, 3)
    omega_inertial_history = np.array(sat_ang_vel)
    angular_momentum_inertial_history = np.array(sat_ang_mom)

    # Calculate the plane of the herpolhode
    # Get the constant from the first time step
    herpolhode_const = np.dot(angular_momentum_inertial_history[0, :], omega_inertial_history[0, :])
    print(f"Herpolhode constant for {sat_name} is {herpolhode_const}")

    # Compare to principal frame
    herpolhode_const_principal = np.dot(initial_angular_momentum, initial_omega)
    print(f"Herpolhode constant for principal frame is {herpolhode_const_principal}")

    herpolhode_by_time = np.sum(angular_momentum_inertial_history * omega_inertial_history, axis=1)
    plt.plot(sat_times, herpolhode_by_time, label=f"{sat_name} herpolhode",
                linestyle="--" if sat_name == "MRP" else ":")
    
plt.xlabel("Time [s]")
plt.ylabel("Herpolhode Constant (L dot omega) [kg m^2/s^2]")
plt.legend()
plt.grid()
plt.show()

# Save the figure
fig.savefig(f"ps3_data/herpolhode_constant_max{max_i}_skip{num_skip}.png")
