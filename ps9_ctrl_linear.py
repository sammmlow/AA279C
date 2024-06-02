# -*- coding: utf-8 -*-

import datetime
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

from source import perturbations
from source.spacecraft import Spacecraft
from source.attitudes import QTR, MRP
from source.rotation import dcmX, dcmY, dcmZ

from plot_everything import plot_everything

np.random.seed(seed=1)

file_path = "figures/ps9/PS9-LinCtrlOnly-"

plt.close("all")

current_time = datetime.datetime(2025, 1, 1, 12, 0, 0) # For ECEF computation.

# ===========================================================================
# Toggle these flags to determine the fate of this script.
# ===========================================================================

bool_enable_perturbations = True
bool_enable_active_control = True
bool_plot_orbit = True

# ===========================================================================
# Parameters for linearized PID controller
# ===========================================================================

Kp = 50 * np.array([-5.24E-5, -3.99E-5, -3.37E-5])
Kd = 50 * np.array([-1.0, -1.0, -1.0])

# ===========================================================================
# Initializing the spacecraft and dynamics environment...
# ===========================================================================

sc = Spacecraft( elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6],
                 inertia = np.diag( [4770.398, 6313.894, 7413.202] ) )

# Get the spacecraft into RTN configuration.
initial_omega = np.array([0, 0, sc.n])
initial_dcm = sc.get_hill_frame().T  # RTN2ECI

# Body (B) to inertial (N) angular velocity and attitude
sc.ohmBN = -initial_omega
sc.attBN = QTR( dcm = initial_dcm )


# ===========================================================================
# Simulation time and sampling parameters...
# ===========================================================================

period = 86400
number_of_orbits = 5
duration = number_of_orbits * period
now, n, timestep = 0.0, 0, 120.0
nBig = 0  # Just a counter for plotting attitude triads in 3D...

samples = int(duration / timestep)
sample_bigstep = 6
sample_trigger_interval = duration / sample_bigstep
skip = 5 # Larger skip leads to faster plotting
sample_trigger_count = 0.0

timeAxis = np.linspace(0, duration, samples)

print("Number of steps: ", samples)
print("Sample skip rate: ", skip)
print("Number of plotted samples: ", samples // skip)


# ===========================================================================
# Containers for storing states for plotting later on...
# ===========================================================================

states_pos   = np.zeros(( 3, samples ))
states_angle = np.zeros(( 3, samples ))
states_ohmBN = np.zeros(( 3, samples ))
states_ohmBR = np.zeros(( 3, samples ))
states_qtrBN = np.zeros(( 4, samples ))
states_qtrBR = np.zeros(( 4, samples ))
states_gtorq = np.zeros(( 3, samples ))
states_mtorq = np.zeros(( 3, samples ))
states_storq = np.zeros(( 3, samples ))

states_ctrl = np.zeros(( 3, samples ))

states_pos_sampled = np.zeros(( 3, samples ))
states_dcm_sampled = np.zeros(( 3, 3, samples ))


# ===========================================================================
# Actual dynamics simulation below.
# ===========================================================================

while now < duration:

    # Store spacecraft states.
    states_pos[:, n] = sc.states[0:3]
    states_ohmBN[:, n] = sc.ohmBN
    states_qtrBN[:, n] = sc.attBN.qtr
    states_angle[:, n] = sc.attBN.get_euler_angles_321()

    # Compute reference-to-inertial omegas and attitudes.
    ohmRN = -initial_omega
    attRN = QTR( dcm = sc.get_hill_frame().T ) # RTN2ECI

    # Compute body-to-reference (controller error) omegas and attitudes.
    ohmBR = sc.ohmBN - ohmRN
    qtrBR = sc.attBN / attRN # Quaternion multiplication.

    # Represent as short-only rotation for qtrBR.
    # Very important for the controller!
    if qtrBR[0] < 0.0:
        qtrBR.qtr = -1 * qtrBR.qtr

    # Store body-to-reference (controller error) omegas and attitudes.
    states_ohmBR[:, n] = ohmBR
    states_qtrBR[:, n] = qtrBR

    # Sample DCM for plotting triads later on.
    if (now >= sample_trigger_count):
        states_pos_sampled[:, nBig] = sc.states[0:3]
        states_dcm_sampled[:, :, nBig] = sc.attBN.dcm
        sample_trigger_count += sample_trigger_interval
        nBig += 1

    # Initialize total torques
    pert_torque = np.zeros(3)
    ctrl_torque = np.zeros(3)

    if bool_enable_perturbations:

        # Compute gravity gradient perturbation torques.
        gTorque = perturbations.compute_gravity_gradient_torque(
            sc.GM, sc.attBN.dcm.T @ sc.states[0:3], sc.inertia)

        # Compute magnetic field perturbation torques.
        mTorque = perturbations.compute_magnetic_torque_component(
            current_time, sc.states[0:3], sc.attBN)

        # Compute solar radiation pressure perturbation torques.
        sTorque = perturbations.compute_solar_torque_component(
            current_time, sc.states[0:3], sc.attBN)

        # Add to total torques
        pert_torque += (gTorque + mTorque + sTorque)

        # Store the computed perturbation torques.
        states_gtorq[:, n] = gTorque
        states_mtorq[:, n] = mTorque
        states_storq[:, n] = sTorque

    # Compute controller torque using state feedback. Use MRPs.
    if bool_enable_active_control:
        mrpBR = MRP( dcm = qtrBR.dcm )
        mrpBR_dot = mrpBR.get_mrpRate( ohmBR )
        ctrl_torque = Kp * mrpBR.mrp + Kd * mrpBR_dot
        # ctrl_gyro_terms = 2 * np.cross(ohmBR, sc.inertia) @ sc.ohmBN
    
    # Store control torque.
    states_ctrl[:, n] = ctrl_torque
    
    # Propagate the attitude and the angular velocity
    sc.propagate_orbit(timestep)
    sc.propagate_attitude(timestep, torque = pert_torque + ctrl_torque )

    # Update simulation time and calendar time
    current_time = current_time + datetime.timedelta(seconds = timestep)
    now += timestep
    n += 1

# ===========================================================================
# Plot everything!
# ===========================================================================

nil = np.array([])

plot_everything( timeAxis, skip, period, number_of_orbits, file_path,
                  states_qtrBN, nil, nil, nil, states_angle, states_ohmBN,
                  states_pos, states_pos_sampled, states_dcm_sampled,
                  states_qtrBR, states_ohmBR, bool_plot_orbit, states_ctrl )
