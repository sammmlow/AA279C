################
# Test the Actuator class with reaction wheels and thrusters

import numpy as np
import matplotlib.pyplot as plt

from source.actuators import Actuator

# Plot saving
file_path = "figures/ps9/PS9-ACTUATOR-CLASS-"
dpi=200
bbox_inches='tight'

# Load the mounting matrices from file.
a_rw = np.loadtxt("rom/ps9_actuator_reaction_wheel_mounting_matrix.csv", delimiter=",")
a_thr = np.loadtxt("rom/ps9_actuator_thruster_mounting_matrix.csv", delimiter=",")

# Reaction wheel limits
max_torque_rw = 0.25 # Nm
min_torque_rw = -0.25 # Nm
noise_std_rw = 0.01 # Nm

# Thruster limits
max_thrust_thr = 1.1 # N
min_thrust_thr = 0.0 # N
noise_std_thr_N = 0.1 # N

# Create the actuator objects
actuator_rw = Actuator(a_rw, noise_std_rw, max_torque_rw, min_torque_rw)
actuator_thr = Actuator(a_thr, noise_std_thr_N, max_thrust_thr, min_thrust_thr)

# Convert the thrust limits to Nm with the moment arm from the mounting matrix
# print(a_thr)
# print(a_thr.shape)
# max_torque_thr = a_thr @ np.array([max_thrust_thr, 0, 0])
# min_torque_thr = a_thr @ np.array([min_thrust_thr, 0, 0])
# noise_std_thr = a_thr @ np.array([noise_std_thr_N, 0, 0])

# Test the reaction wheel actuator
# Slowly ramp up the commanded torques
n_steps = 100
t_steps = np.linspace(0, 1, n_steps)
mc_commanded = t_steps * max_torque_rw * 16
mc_vectors = [np.array([1, 0, 0]), np.array([0, 1, -2]), np.array([0, -2, 5]),
              np.array([1, 2, 3])]

# Normalize the mc_vectors
mc_vectors = [mc_vec / np.linalg.norm(mc_vec) for mc_vec in mc_vectors]

# Plot the results
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))

legend_handles = []
legend_handles_done = False

for ax_ind, mc_vector in enumerate(mc_vectors):
    mc_commanded_vec = np.outer(mc_commanded, mc_vector)
    assert mc_commanded_vec.shape == (n_steps, 3)

    actuator_applied_hist = np.zeros((n_steps, actuator_rw.n_actuators))
    actuator_command_hist = np.zeros((n_steps, actuator_rw.n_actuators))

    for idx in range(n_steps):
        actuator_applied, actuator_command = \
            actuator_rw.calculate_actuator_commands(mc_commanded_vec[idx] * np.sin(idx/10))
        actuator_applied_hist[idx, :] = actuator_applied
        actuator_command_hist[idx, :] = actuator_command

    # Get the default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ax_curr = ax[ax_ind // 2, ax_ind % 2]

    # We plot the commanded torques as dashed (i.e., not realized) and the applied
    # torques as solid (i.e., realized).
    for actuator_idx in range(actuator_rw.n_actuators):
        actuator_line = ax_curr.plot(
            t_steps, actuator_command_hist[:, actuator_idx],
            label=f"Actuator {actuator_idx+1} [Commanded]", linestyle="--",
            color=colors[actuator_idx], zorder=1)
        command_line = ax_curr.plot(
            t_steps, actuator_applied_hist[:, actuator_idx],
            label=f"Actuator {actuator_idx+1} [Applied]",
            color=colors[actuator_idx], zorder=2)
        
        if not legend_handles_done:
            legend_handles.append(actuator_line[0])
            legend_handles.append(command_line[0])

    ax_curr.set_ylabel(f"Angular Momentum Change (Nm)")
    ax_curr.set_xlabel(f"Simulation Time (s)")

    ax_curr.set_ylim(-1.2, 1.2)

    # Add the vector to the plot as text in the top left
    ax_curr.text(0.01, 0.95, 
        f"MC Vector: [{mc_vector[0]:.2f}, {mc_vector[1]:.2f}, {mc_vector[2]:.2f}]")

    legend_handles_done = True
    # ax_curr.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.figlegend(legend_handles, [h.get_label() for h in legend_handles],
                loc='center left', bbox_to_anchor=(0.9, 0.5))

# plt.tight_layout()
# plt.show()
plt.savefig(file_path + "reaction_wheel_actuator_commands.png",
            dpi=dpi, bbox_inches=bbox_inches)
