# -*- coding: utf-8 -*-

# Part 1: Assume that 2 components of the initial angular velocities are zero,
# and that the principal axes are aligned with the inertial frame (e.g., zero
# Euler angles). Verify that during the simulation the 2 components of angular
# velocity remain zero, and that the attitude represents a pure rotation about
# the rotation axis (e.g., linearly increasing Euler angle). Plot velocities
# and angles.

import numpy as np
import datetime
import matplotlib.pyplot as plt

from source import perturbations
from source.rotation import dcmX, dcmY, dcmZ
from source.spacecraft import Spacecraft
from source.attitudes import QTR, MRP
from source.plot_orbit_and_attitude import plot_orbit_and_attitude

from plot_everything import plot_everything

# ===========================================================================

# For saving the figures
file_path = "figures/ps5/PS5-Pert-StabilityTests-"

# Assume a current time, simply for calculating ECEF coordinates.
current_time = datetime.datetime(2025, 1, 1, 12, 0, 0)

plt.close("all")

initial_inertia = np.diag( [4770.398, 6313.894, 7413.202] );
    
geo_elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6];
sc = Spacecraft( elements = geo_elements )

# Setup the inertias, omegas, and initial attitudes... note the order of
# operations of rotations! Perturb the RTN first, then snap the full
# rotation to the perturbed RTN frame.
dcm_rotation = dcmY( np.deg2rad(-90.0) )
dcm_rtn =  sc.get_hill_frame().T
initial_omega = np.array([0, 0, sc.n]);

# Set the initial omegas, attitudes, and inertias.
# Perturb by +1 deg in all directions.
sc.ohmBN = dcm_rotation @ initial_omega
sc.attBN = QTR( dcm = dcm_rotation @ dcm_rtn );
sc.inertia = initial_inertia

# Initialize simulation time parameters.
# Orbit period is 24 hours, or 60 s/m * 60 m/hr * 24hr/day.
one_orbital_period = 60 * 60 * 24
n_periods = 10
duration = n_periods * one_orbital_period

now, n, timestep = 0.0, 0, 60.0
samples = int(duration / timestep)
print("Number of samples: ", samples)
timeAxis = np.linspace(0, duration, samples)
# sample_bigstep = 8
sample_bigstep = 11
sample_trigger_interval = duration / sample_bigstep

# Initialize containers for plotting.
states_pos   = np.zeros(( 3, samples ))
states_omega = np.zeros(( 3, samples ))
states_angle = np.zeros(( 3, samples ))
states_quatr = np.zeros(( 4, samples ))
states_gtorq = np.zeros(( 3, samples ))
states_mtorq = np.zeros(( 3, samples ))
states_storq = np.zeros(( 3, samples ))
states_pos_sampled = np.zeros(( 3, samples ))
states_dcm_sampled = np.zeros(( 3, 3, samples ))

# Just a counter for plotting the number of attitude triads in the 3D plot.
nBig = 0

# Make this number bigger to plot faster with fewer points.
skip = 5
# sampleSkip = 40
print(f"with skip of {skip}, number of samples: ", 
      samples // skip)

sample_trigger_count = 0.0;

# Propagate attitude and orbit
while now < duration:
    
    # Store the angular velocities and 321 Euler angles
    states_pos[:, n] = sc.states[0:3]
    states_omega[:, n] = sc.ohmBN
    states_angle[:, n] = sc.attBN.get_euler_angles_321()
    states_quatr[:, n] = sc.attBN.qtr
    
    # Fragile code. Will not work if time step skips this.
    if (now >= sample_trigger_count):
        states_pos_sampled[:, nBig] = sc.states[0:3]
        states_dcm_sampled[:, :, nBig] = sc.attBN.dcm
        sample_trigger_count += sample_trigger_interval
        nBig += 1
        
    # Compute gravity gradient perturbation torques.
    gTorque = perturbations.compute_gravity_gradient_torque(
        sc.GM, sc.attBN.dcm.T @ sc.states[0:3], sc.inertia)
    
    # Compute magnetic field perturbation torques.
    mTorque = perturbations.compute_magnetic_torque_component(
        current_time, sc.states[0:3], sc.attBN)
    
    # Compute solar radiation pressure perturbation torques.
    sTorque = perturbations.compute_solar_torque_component(
        current_time, sc.states[0:3], sc.attBN)
    
    # Store the computed perturbation torques
    states_gtorq[:, n] = gTorque
    states_mtorq[:, n] = mTorque
    states_storq[:, n] = sTorque
    
    # Propagate the attitude and the angular velocity
    sc.propagate_orbit(timestep)
    sc.propagate_attitude(timestep, torque = gTorque + mTorque + sTorque)
    
    now += timestep
    n += 1
    
    # Update the actual calendar time
    current_time = current_time + datetime.timedelta(seconds = timestep)
    
    
## ===========================================================================
## PLOTTING STUFF! KEEP YOUR EYES AWAY!
## ===========================================================================
    
plot_everything( timeAxis, skip, one_orbital_period, n_periods, file_path,
                     states_quatr, states_gtorq, states_mtorq,
                     states_storq, states_angle, states_omega,
                     states_pos, states_pos_sampled, states_dcm_sampled )
