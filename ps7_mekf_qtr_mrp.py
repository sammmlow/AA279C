# -*- coding: utf-8 -*-

import datetime
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

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
    S13 = np.zeros((3,3))
    S21 = np.zeros((3,3))
    S22 = eye3 + F_ww * dt
    S23 = np.zeros((3,3))
    S31 = np.zeros((3,3))
    S32 = np.zeros((3,3))
    S33 = eye3
    return np.block([[S11, S12, S13], [S21, S22, S23], [S31, S32, S33]])

# Input is a spacecraft object (used as a data structure for convenience)
def mekf_time_update(
        sc_model, dt, ctrl_torque, modelled_pert_torque, cov, Q):
    initial_omega = sc_model.ohmBN
    stm = compute_stm_mekf(initial_omega, sc_model.inertia, dt)
    final_omega = stm[3:6,3:6] @ initial_omega
    sc_model.propagate_attitude( dt, ctrl_torque )
    sc_model.ohmBN = final_omega
    updated_cov = stm @ cov @ stm.T + Q
    return [sc_model, updated_cov]

# Input is a spacecraft object (used as a data structure for convenience).
# For computational efficiency, we call measurement update per measurement.
# "dcm_S2B" is the sensor to body frame DCM, assumed to be known from a CAD.
# this function assumes we have both star tracker and gyro information at 
# the same time stamp.
def mekf_meas_update(
        sc_model, mean, cov, yMeas_S, yModel_N, wMeas_B, dcm_S2B, R):
    
    # Compute modelled measurements for unit vectors
    dcm_N2B = sc_model.attBN.dcm.T
    yMeas_B = dcm_S2B @ yMeas_S
    yModel_B = dcm_N2B @ yModel_N
    
    # Compute modelled measurements for angular velocities
    # wMeas_B is given as an input to the function
    wBias_B = mean[6:9]
    wModel_B = sc_model.ohmBN + wBias_B # Modelled meas uses bias est
    
    # Compute the sensitivity model (SUPER DUPER CREDITS TO DANIEL NEAMATI)
    H_mrp = 4 * make_skew_symmetric(yModel_B)
    H_eye = np.identity(3)
    H_zeros = np.zeros((3,3))
    H = np.block([[H_mrp, H_zeros, H_zeros], [H_zeros, H_eye, H_eye]])
    
    # Full residual vector
    resd_i = np.concatenate([ (yMeas_B - yModel_B), (wMeas_B - wModel_B) ])
    prefit = np.array([ norm(yMeas_B - yModel_B), norm(wMeas_B - wModel_B) ])
    
    # Kalman gain, state mean and cov update.
    K = cov @ H.T @ np.linalg.inv(H @ cov @ H.T + R)
    updated_mean = mean + K @ resd_i
    updated_cov = cov - K @ H @ cov
    
    # Update the quaternion in the spacecraft model by correcting with dMRP.
    mrps = MRP( mrp = updated_mean[0:3] )
    updated_qtr = QTR( dcm = mrps.dcm @ sc_model.attBN.dcm )
    sc_model.attBN = updated_qtr
    
    # Generate postfit measurements now
    post_dcm_N2B = sc_model.attBN.dcm.T
    post_yModel_B = post_dcm_N2B @ yModel_N
    post_wBias_B = updated_mean[6:9]
    post_wModel_B = sc_model.ohmBN + post_wBias_B
    
    postfit = np.array([ norm(yMeas_B - post_yModel_B),
                         norm(wMeas_B - post_wModel_B) ])
    
    return [sc_model, updated_mean, updated_cov, prefit, postfit]

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
history_est_ohm_bias = np.zeros(( 3, samples ))  # Part of the state

prefit_samples = np.zeros(( 2, 10 * samples ))
posfit_samples = np.zeros(( 2, 10 * samples ))

covariance = np.zeros(( 9, 9, samples + 1 ))  # Full history

# Initialize some starting covariance
init_covariance = np.diag([1E-7] * 9)
covariance[:, :, 0] = init_covariance

# Some arbitrary process noise?
Q = np.diag([1E-8] * 9)

# Some arbitrary measurement noise? Match the noise source below.
R = np.diag([0.001 * np.pi / 180.0] * 3 + [0.00001] * 3)

# Create a state that keeps track of estimated omega biases (because the
# current spacecraft object doesn't have a way to store it right now...)
est_ohm_bias = np.zeros(3)

# ===========================================================================
# Function that generates a noisy DCM rotation.
# ===========================================================================

from source.rotation import dcmX, dcmY, dcmZ

noise_deg = 0.1 # degrees
noise_rad = 0.1 * np.pi / 180.0

omega_bias = np.zeros(3) # 0.0001 * np.ones(3) # rad/s

def make_a_noisy_dcm():
    x = dcmX( np.random.normal(0.0, noise_rad) )
    y = dcmY( np.random.normal(0.0, noise_rad) )
    z = dcmZ( np.random.normal(0.0, noise_rad) )
    return x @ y @ z

# Generate some fake sensor mounting matrix.
dcm_S2B = dcmX(0.1) @ dcmY(0.1) @ dcmZ(0.1)

# Make a copy of the actual spacecraft. This will be our "model" spacecraft
# which will simulate the spacecraft ADCS receiving noisy measurements.
# The original `sc` will remain as the ground truth.
sc_model = Spacecraft( elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6],
                       inertia = np.diag( [4770.398, 6313.894, 7413.202] ) )

sc_model.ohmBN = -initial_omega + 0.000001 * np.random.rand(3)
sc_model.attBN = QTR( dcm = initial_dcm ) # Absolute state

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
    
    # Generate a bunch of noisy ground truth measurements. For simplicity,
    # we'll just generate an omega measurement for each star tracker meas.
    k = 0
    for star in catalog:
        
        true_star_N = star[:3]
        true_omega_BN = sc.ohmBN
        
        noisy_omega_BN = make_a_noisy_dcm() @ true_omega_BN + omega_bias
        noisy_star_S = make_a_noisy_dcm() @ true_star_N
        
        current_mean_mrp = np.zeros(3)
        current_mean_ohm = sc_model.ohmBN
        current_mean_bias = est_ohm_bias
        
        current_mean = np.concatenate([
            current_mean_mrp,
            current_mean_ohm,
            current_mean_bias])
        
        # Perform the MEKF measurement update.
        [sc_model, updated_mean, updated_cov, pre, post] = mekf_meas_update(
            sc_model,
            current_mean,
            current_cov,
            noisy_star_S,
            true_star_N,
            noisy_omega_BN,
            dcm_S2B,
            R)
        
        # Save prefit and postfit samples
        prefit_samples[:, 10 * n + k] = pre
        posfit_samples[:, 10 * n + k] = post
        k += 1
        
    
    # At this point, extract out the error MRPs for plotting.
    current_mean_mrp = updated_mean[0:3]
    current_mean_ohm = updated_mean[3:6]
    est_ohm_bias = updated_mean[6:9]
    
    sc_model.ohmBN = current_mean_ohm
    
    # Update the saved states for plotting later on.
    errors_ohmBN[:, n] = current_mean_ohm - sc.ohmBN
    errors_qtrBN[:, n] = sc_model.attBN.qtr - sc.attBN.qtr
    errors_mrpErrorBN[:, n] = current_mean_mrp
    history_est_ohm_bias[:, n] = est_ohm_bias
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

print("Plotting angular velocity bias estimation")
fig1b, axes1b = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
labels = ['$\omega_{b,x}$', '$\omega_{b,y}$', '$\omega_{b,z}$']
for i, ax in enumerate(axes1b):
    ax.plot( timeAxis[::skip], history_est_ohm_bias[i,::skip])
    ax.fill_between( timeAxis,
                     np.sqrt(covariance[6+i,6+i,:-1]),
                     -np.sqrt(covariance[6+i,6+i,:-1]),
                     alpha=0.2)
    ax.set_ylabel(labels[i] + ' [rad/s]')
    ax.grid(True)
    ax.axhline(omega_bias[i], color='k', linestyle='-')
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
    ax.plot( timeAxis[::skip], errors_mrpErrorBN[i,::skip])
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


# Plot prefit and postfit samples for MRPs
plt.figure()
plt.plot(prefit_samples[0,:])
plt.plot(posfit_samples[0,:])
plt.grid('on')
plt.xlabel('Samples'); plt.ylabel('Star tracker residuals (unitless)')
plt.legend(["Prefit", "Postfit"])
plt.savefig(file_path + 'ResdStar.png', dpi=200, bbox_inches='tight')

# Plot prefit and postfit samples for omegas
plt.figure()
plt.plot(prefit_samples[1,:])
plt.plot(posfit_samples[1,:])
plt.grid('on')
plt.xlabel('Samples'); plt.ylabel('Angular velocity meas residuals (rad/s)')
plt.legend(["Prefit", "Postfit"])
plt.savefig(file_path + 'ResdOmega.png', dpi=200, bbox_inches='tight')