# -*- coding: utf-8 -*-

import datetime
import numpy as np
import matplotlib.pyplot as plt

file_path = "figures/ps7/PS7-Errors-WithPertModelled-"

# Copy over the star catalog into a matrix.
catalog = np.genfromtxt(
    "hipparcos_star_catalog_solutions_downselected_10.txt", delimiter=' ')

# This script is an attempt at creating an MEKF routine using quaternions as
# the absolute attitude and MRPs as the delta error attitude to be updated
# during the measurement update. The rationale is that MRPs are "twice" as
# linearizable than quaternions, and hopefully a linear measurement update
# would be less erroneous for MRPs than quaternions. Oh well, here we go!

from source import perturbations
from source.spacecraft import Spacecraft
from source.attitudes import QTR, MRP

def make_skew_symmetric(v):
    vSkew = np.array([[0.0, -1*v[2], v[1]],
                      [v[2], 0.0, -1*v[0]],
                      [-1*v[1], v[0], 0.0]])
    return vSkew

# Computes the simplified STM for MRPs using MEKF framework where MRPs = 0
def compute_stm_mekf(w, I, dt):
    
    eye3 = np.identity(3)
    wSkew = make_skew_symmetric(w)
    wx, wy, wz = w[0], w[1], w[2]
    Ix, Iy, Iz = I[0,0], I[1,1], I[2,2]
    
    F_ww = np.array([[0.0, wz*(Iy-Iz)/Ix, wy*(Iy-Iz)/Ix],
                     [wz*(Iz-Ix)/Iy, 0.0, wx*(Iz-Ix)/Iy],
                     [wy*(Ix-Iy)/Iz, wx*(Ix-Iy)/Iz, 0.0]])
    
    S11 = eye3 - 0.5 * wSkew * dt
    S12 = 0.25 * eye3 * dt
    S21 = np.zeros((3,3))
    S22 = eye3 + F_ww * dt
    return np.block([[S11, S12], [S21, S22]])

# Input is a spacecraft object (used as a data structure for convenience)
def mekf_time_update(sc_model, dt, ctrl_torque, modelled_pert_torque, cov, Q):
    stm = compute_stm_mekf(sc_model.ohmBN, sc_model.inertia, dt)
    initial_omega = sc.ohmBN
    final_omega = stm[3:,3:] @ initial_omega
    sc_model.propagate_attitude( dt, ctrl_torque + modelled_pert_torque )
    # Note that propagate_attitude also updates the omega but we want to use
    # the linearized update to simulate the discretized dynamics; it's silly.
    sc_model.ohmBN = final_omega
    updated_cov = stm @ cov @ stm.T + Q
    return [sc_model, updated_cov]

# ===========================================================================
# Will be using the spacecraft as a container for my "state vector"...
# ===========================================================================

sc = Spacecraft( elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6],
                 inertia = np.diag( [4770.398, 6313.894, 7413.202] ) )

# Get the spacecraft into RTN configuration.
initial_omega = np.array([0, 0, sc.n])
initial_dcm = sc.get_hill_frame().T  # RTN2ECI

# Body (B) to inertial (N) angular velocity and attitude
sc.ohmBN = -initial_omega
sc.attBN = QTR( dcm = initial_dcm ) # Absolute state

# Make a copy of the actual spacecraft. This will be our "model" spacecraft
# which will simulate the spacecraft ADCS receiving noisy measurements.
# The original `sc` will remain as the ground truth.
sc_model = Spacecraft( elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6],
                       inertia = np.diag( [4770.398, 6313.894, 7413.202] ) )
sc_model.ohmBN = -initial_omega + 0.000001 * np.random.rand(3)
sc_model.attBN = QTR( dcm = initial_dcm ) # Absolute state


# ===========================================================================
# Simulation time and sampling parameters...
# ===========================================================================

current_time = datetime.datetime(2025, 1, 1, 12, 0, 0) # For ECEF computation.
plt.close("all")

period = 86400
number_of_orbits = 1
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

errors_ohmBN      = np.zeros(( 3, samples ))  # Part of the state
errors_qtrBN      = np.zeros(( 4, samples ))  # NOT part of the state
errors_mrpErrorBN = np.zeros(( 3, samples ))  # Part of the state

covariance = np.zeros(( 6, 6, samples + 1 ))  # Full history

# Initialize some starting covariance
init_covariance = np.diag([1E-12] * 6)

# Some arbitrary process noise?
Q = np.diag([1E-13] * 6)

# ===========================================================================
# Actual dynamics simulation below.
# ===========================================================================

while now < duration:
    
    # Initialize total torques
    pert_torque = np.zeros(3)
    ctrl_torque = np.zeros(3)
        
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
    
    # Propagate the attitude and the angular velocity
    sc.propagate_orbit(timestep)
    sc.propagate_attitude(timestep, torque = pert_torque + ctrl_torque )
    
    # Perform the MEKF time update
    current_cov = covariance[:, :, n]
    [sc_model, updated_cov] = mekf_time_update(
        sc_model, timestep, ctrl_torque, pert_torque, current_cov, Q)
    
    # Perform the MEKF measurement update
    # TODO. Also, at this point, extract out the error MRPs for plotting.
    
    # Update the reference quaternion attitude with the MRPs.
    
    
    # Update the saved states for plotting later on.
    errors_ohmBN[:, n] = sc_model.ohmBN - sc.ohmBN
    errors_qtrBN[:, n] = sc_model.attBN.qtr - sc.attBN.qtr
    errors_mrpErrorBN[:, n] = np.zeros(3) # Requires meas update.
    covariance[:, :, n+1] = updated_cov
    
    # Update simulation time and calendar time
    current_time = current_time + datetime.timedelta(seconds = timestep)
    now += timestep
    n += 1

# Plot everything

print("Plotting angular velocity errors")
fig1, axes1 = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
labels = ['$\omega_x$', '$\omega_y$', '$\omega_z$']
for i, ax in enumerate(axes1):
    ax.plot( timeAxis[::skip], errors_ohmBN[i,::skip])
    ax.fill_between( timeAxis,
                     np.sqrt(covariance[3+i,3+i,:-1]),
                     -np.sqrt(covariance[3+i,3+i,:-1]),
                     alpha=0.2)
    ax.set_ylabel(labels[i] + ' [rad/s]')
    ax.grid(True)
    if i == 2:
        ax.set_xlabel('Time [seconds]')
    for i in range(number_of_orbits + 1):
        ax.axvline(i * period, color='gray', linestyle='--')
plt.show()
plt.savefig(file_path + 'Omegas.png', dpi=200, bbox_inches='tight')

print("Plotting error of the updated reference quaternions")
fig2, axes2 = plt.subplots(nrows=4, ncols=1, figsize=(7, 6))
labels = ['$q_0$', '$q_1$', '$q_2$', '$q_3$']
for i, ax in enumerate(axes2):
    ax.plot( timeAxis[::skip], errors_qtrBN[i,::skip])
    ax.set_ylabel(labels[i])
    ax.grid(True)
    if i == 3:
        ax.set_xlabel('Time [seconds]')
    for i in range(number_of_orbits + 1):
        ax.axvline(i * period, color='gray', linestyle='--')
plt.show()
plt.savefig(file_path + 'QTR.png', dpi=200, bbox_inches='tight')

print("Plotting estimated error MRP")
fig3, axes3 = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
labels = ['$\sigma_1$', '$\sigma_2$', '$\sigma_3$']
for i, ax in enumerate(axes3):
    ax.plot( timeAxis[::skip], errors_ohmBN[i,::skip])
    ax.set_ylabel(labels[i])
    ax.fill_between( timeAxis,
                     np.sqrt(covariance[i,i,:-1]),
                     -np.sqrt(covariance[i,i,:-1]),
                     alpha=0.2)
    ax.grid(True)
    if i == 2:
        ax.set_xlabel('Time [seconds]')
    for i in range(number_of_orbits + 1):
        ax.axvline(i * period, color='gray', linestyle='--')
plt.show()
plt.savefig(file_path + 'ErrMRP.png', dpi=200, bbox_inches='tight')