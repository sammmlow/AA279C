# -*- coding: utf-8 -*-

# ===========================================================================
# State-transition matrix: Sub-blocks for each STM
# ===========================================================================

import datetime
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

from source import perturbations
from source.spacecraft import Spacecraft
from source.attitudes import QTR, MRP
from source.rotation import dcmX, dcmY, dcmZ

file_path = "figures/ps9/PS9-MEKF-"

# Copy over the star catalog into a matrix.
nStars = 10 # Hardcoded.
catalog = np.genfromtxt(
    "hipparcos_star_catalog_solutions_downselected_10.txt", delimiter=' ')

# ===========================================================================
# Constants used throughout the simulation
# ===========================================================================

# Principal inertia tensor of the spacecraft.
inertia = np.diag([ 4770.398, 6313.894, 7413.202 ])

# Arbitrary sensor mounting rotation matrix from sensor frame to body frame.
DCM_S2B = dcmX(0.1) @ dcmY(0.1) @ dcmZ(0.1)
DCM_B2S = DCM_S2B.T

# ===========================================================================
# State-transition matrix: Sub-blocks for each STM
# ===========================================================================

I3 = np.identity(3)
Z3 = np.zeros((3,3))

# Create a skew symmetric matrix
def skew( v ):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

# Create the STM for the following 1x9 states: [MRPs, Omegas, OmegaBias]
def compute_stm( mean, dt ):
    
    wx, wy, wz = mean[3], mean[4], mean[5] # Biased angular velocity state
    bx, by, bz = mean[6], mean[7], mean[8] # Gyroscopic bias states
    
    Ia = (inertia[1,1] - inertia[2,2]) / inertia[0,0]
    Ib = (inertia[2,2] - inertia[0,0]) / inertia[1,1]
    Ic = (inertia[0,0] - inertia[1,1]) / inertia[2,2]
    
    F22 = np.array([[0, wz*Ia, wy*Ia], [wz*Ib, 0, wx*Ib], [wy*Ic, wx*Ic, 0]])
    F23 = np.array([[0, bz*Ia, by*Ia], [bz*Ib, 0, bx*Ib], [by*Ic, bx*Ic, 0]])
    
    S11 = I3 - 0.5 * skew(mean[3:6]) * dt
    S12 = 0.25 * I3 * dt
    S13 = 0.25 * I3 * dt
    S21 = Z3
    S22 = I3 + F22
    S23 = F23
    S31 = Z3
    S32 = Z3
    S33 = I3
    
    return np.block([[S11, S12, S13], [S21, S22, S23], [S31, S32, S33]])
    

# ===========================================================================
# Multiplicative EKF: Time Update Functions
# ===========================================================================

# Reference quaternions are propagated using the Spacecraft class.
def propagate_quaternion( dt, omega, quaternion, torque ):
    
    model = Spacecraft()
    model.inertia = inertia
    model.attBN = quaternion
    model.ohmBN = omega
    model.propagate_attitude( dt, torque )
    
    return model.attBN

# MEKF time update for 1x9 states: [MRPs, Omegas, OmegaBias]
def time_update( dt, mean, covariance, quaternion, torque, Q ):
    
    omega = mean[3:6] # Angular velocity body-to-inertial.
    STM = compute_stm( mean, dt )
    initial_mean = np.concatenate([ [0,0,0], mean[3:9] ])
    updated_mean = STM @ initial_mean
    updated_cov = STM @ covariance @ STM.T + Q
    updated_quaternion = propagate_quaternion( dt, omega, quaternion, torque )
    
    return [updated_mean, updated_cov, updated_quaternion]


# ===========================================================================
# Multiplicative EKF: Measurement Update Function.
# 
# Assumes to receive a set of star tracker measurements as noisy unit
# direction vectors, and a single biased angular velocity measurement
# directly from the gyroscope. Note that the catalog of stars is in the
# inertial (N) frame, and the measurements are in the sensor (S) frame.
# Angular velocities are body-to-inertial (BN).
# ===========================================================================

def meas_update( mean, cov, quat, catalog, meas_stars, meas_omega, R ):
    
    # Check the number of stars in `meas_stars` tally with catalog.
    assert (np.shape(catalog)[1] == 4
            ), "Check catalog dimensions!"
    assert (np.shape(meas_stars)[1] == 4
            ), "Check measurements dimensions!"
    assert (np.shape(catalog)[0] == np.shape(meas_stars)[0]
            ), "Mismatched number of stars between catalog and measurements!"
    N = np.shape(meas_stars)[0]
    
    # Transform true (S-frame) and modelled measurements (N-frame) from the
    # star tracker into a common body reference frame.
    DCM_N2B = quat.dcm.T
    meas_stars = DCM_S2B @ meas_stars[:, 0:3].T
    model_stars = DCM_N2B @ catalog[:, 0:3].T
    
    # Compute modelled measurements for angular velocities, including bias.
    model_omega = mean[3:6] + mean[6:9]
    
    # Compute the sensitivity matrix with dimensions (3n + 3) x 9
    H_mrp = np.zeros((3 * N, 3))
    for n in range(N):
        H_mrp[ (3 * n):(3 * n + 3), 0:3 ] = 4 * skew(model_stars[:, n])
    ZN3 = np.zeros((3 * N, 3))
    H = np.block([[H_mrp, ZN3, ZN3], [Z3, I3, I3]])
    
    # Compute prefit residual L2-norms,
    prefit_stars = norm(meas_stars - model_stars, 2)
    prefit_omega = norm(meas_omega - model_omega, 2)
    prefit = np.array([ prefit_stars, prefit_omega ])
    
    # Full residual vector
    residuals = np.concatenate([ (meas_stars - model_stars).reshape(3 * N),
                                 (meas_omega - model_omega) ])
    
    # Kalman gain, state mean and cov update.
    K = cov @ H.T @ np.linalg.inv(H @ cov @ H.T + R)
    updated_mean = mean + K @ residuals
    updated_cov = cov - K @ H @ cov
    
    # Update the quaternion in the spacecraft model by correcting with dMRP.
    error_mrps = MRP( mrp = updated_mean[0:3] )
    updated_quaternion = QTR( dcm = error_mrps.dcm @ quat.dcm )
    updated_quaternion.normalise()
    
    # Generate postfit measurements now
    DCM_N2B_POST = updated_quaternion.dcm.T
    postfit_model_stars = DCM_N2B_POST @ catalog[:, 0:3].T
    postfit_model_omega = updated_mean[3:6] + updated_mean[6:9]
    postfit = np.array([ norm(meas_stars - postfit_model_stars, 2),
                         norm(meas_omega - postfit_model_omega, 2) ])
    
    return [updated_mean, updated_cov, updated_quaternion, prefit, postfit]


# ===========================================================================
# Functions to generate noisy measurements. Use ground truth to generate them.
# ===========================================================================

noise_rad = 0.00175 # 0.1 degrees
noise_omega = 0.0000175 # 0.001 degrees/s

def make_noisy_dcm():
    x = dcmX( np.random.normal(0.0, noise_rad) )
    y = dcmY( np.random.normal(0.0, noise_rad) )
    z = dcmZ( np.random.normal(0.0, noise_rad) )
    return x @ y @ z

def make_noisy_measurements( catalog, true_quaternion, true_omega, bias ):
    k, meas_stars, meas_omega = 0, catalog, true_omega
    for star in catalog:
        DCM_N2B = true_quaternion.dcm.T
        meas_stars[k, 0:3] = make_noisy_dcm() @ DCM_B2S @ DCM_N2B @ star[:3]
        k += 1
    add_noise_omega = np.random.normal(0.0, noise_omega)
    meas_omega = (make_noisy_dcm() @ meas_omega) + add_noise_omega + bias
    return [meas_stars, meas_omega]


# ===========================================================================
# Set simulation time and sampling parameters...
# ===========================================================================

current_time = datetime.datetime(2025, 1, 1, 12, 0, 0) # For ECEF computation.
plt.close("all")

period = 86400
number_of_orbits = 1
duration = number_of_orbits * period
now, n, timestep = 0.0, 0, 120.0
samples = int(duration / timestep)

skip = 1 # Larger skip leads to faster plotting

timeAxis = np.linspace(0, duration, samples)

print("Number of steps: ", samples)
print("Sample skip rate: ", skip)
print("Number of plotted samples: ", samples // skip)


# ===========================================================================
# Containers for storing states for plotting later on...
# ===========================================================================

errors_mrp  = np.zeros(( 3, samples )) # Part of the state
errors_ohm  = np.zeros(( 3, samples )) # Part of the state
errors_bias = np.zeros(( 3, samples )) # Part of the state
errors_qtr  = np.zeros(( 4, samples )) # NOT part of the state

prefit_samples = np.zeros(( 2, samples ))
posfit_samples = np.zeros(( 2, samples ))

covariance_history = np.zeros(( 9, 9, samples + 1 ))  # Full history


# ===========================================================================
# Initialize ground truth parameters and true spacecraft in RTN configuration.
# ===========================================================================

sc = Spacecraft( elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6],
                 inertia = np.diag([4770.398, 6313.894, 7413.202]) )

initial_omega = np.array([0, 0, sc.n])
initial_dcm = sc.get_hill_frame().T  # RTN2ECI

sc.ohmBN = -initial_omega
sc.attBN = QTR( dcm = initial_dcm )

true_bias = 0.001 * np.ones(3) # rad/s


# ===========================================================================
# Initialize the filter states, covariances, and parameters.
# ===========================================================================

mean = np.zeros(9)
covariance = np.diag([1E-3] * 9)
quaternion = QTR()

Q = np.diag([1E-9] * 9)
R = np.diag([noise_rad] * (3 * nStars) + [noise_omega] * 3)

covariance_history[:, :, 0] = covariance


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
    [mean, cov, quaternion] = time_update(
        timestep, mean, covariance, quaternion, pert_torque + ctrl_torque, Q )
    
    # Generate noisy measurements of stars and angular velocities.
    [meas_stars, meas_omega] = make_noisy_measurements(
        catalog, sc.attBN, sc.ohmBN, true_bias )
    
    # Perform the MEKF measurement update
    [mean, covariance, quaternion, prefit, postfit] = meas_update(
        mean, covariance, quaternion, catalog, meas_stars, meas_omega, R )
        
    # Save prefit and postfit samples
    prefit_samples[:, n] = prefit
    posfit_samples[:, n] = postfit
    
    # Record state errors for plotting later on.
    errors_mrp[:, n]  = mean[0:3]
    errors_ohm[:, n]  = mean[3:6] - sc.ohmBN
    errors_bias[:, n] = mean[6:9] - true_bias
    errors_qtr[:, n]  = quaternion / sc.attBN
    
    # Record covariance in covariance history.
    covariance_history[:, :, n] = covariance
    
    # Update simulation time and calendar time
    current_time = current_time + datetime.timedelta(seconds = timestep)
    now += timestep
    n += 1


# ===========================================================================
# Plot everything.
# ===========================================================================


# Plotting angular velocity errors
print("Plotting angular velocity errors")
fig1, axes1 = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
labels = ['$\omega_x$', '$\omega_y$', '$\omega_z$']
for i, ax in enumerate(axes1):
    ax.plot( timeAxis[::skip], errors_ohm[i,::skip])
    ax.fill_between( timeAxis,
                      np.sqrt(covariance_history[3+i,3+i,:-1]),
                     -np.sqrt(covariance_history[3+i,3+i,:-1]),
                     alpha=0.2)
    ax.set_ylabel(labels[i] + ' [rad/s]')
    ax.grid(True)
    if i == 2:
        ax.set_xlabel('Time [seconds]')
    for i in range(number_of_orbits + 1):
        ax.axvline(i * period, color='gray', linestyle='--')
plt.show()
fig1.savefig(file_path + 'Omegas.png', dpi=200, bbox_inches='tight')


# Plotting angular velocity bias estimation
print("Plotting angular velocity bias estimation")
fig1b, axes1b = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
labels = ['$\omega_{b,x}$', '$\omega_{b,y}$', '$\omega_{b,z}$']
for i, ax in enumerate(axes1b):
    ax.plot( timeAxis[::skip], errors_bias[i,::skip])
    ax.fill_between( timeAxis,
                      np.sqrt(covariance_history[6+i,6+i,:-1]),
                     -np.sqrt(covariance_history[6+i,6+i,:-1]),
                     alpha=0.2)
    ax.set_ylabel(labels[i] + ' [rad/s]')
    ax.grid(True)
    if i == 2:
        ax.set_xlabel('Time [seconds]')
    for i in range(number_of_orbits + 1):
        ax.axvline(i * period, color='gray', linestyle='--')
plt.show()
fig1b.savefig(file_path + 'OmegaBiases.png', dpi=200, bbox_inches='tight')


# Plotting error of the updated reference quaternions
print("Plotting error of the updated reference quaternions")
fig2, axes2 = plt.subplots(nrows=4, ncols=1, figsize=(7, 6))
labels = ['$q_0$', '$q_1$', '$q_2$', '$q_3$']
for i, ax in enumerate(axes2):
    ax.plot( timeAxis[::skip], errors_qtr[i,::skip])
    ax.set_ylabel(labels[i])
    ax.grid(True)
    if i == 3:
        ax.set_xlabel('Time [seconds]')
    for i in range(number_of_orbits + 1):
        ax.axvline(i * period, color='gray', linestyle='--')
plt.show()
fig2.savefig(file_path + 'QTR.png', dpi=200, bbox_inches='tight')


# Plotting estimated error MRP
print("Plotting estimated error MRP")
fig3, axes3 = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
labels = ['$\sigma_1$', '$\sigma_2$', '$\sigma_3$']
for i, ax in enumerate(axes3):
    ax.plot( timeAxis[::skip], errors_mrp[i,::skip])
    ax.set_ylabel(labels[i])
    ax.fill_between( timeAxis,
                      np.sqrt(covariance_history[i,i,:-1]),
                     -np.sqrt(covariance_history[i,i,:-1]),
                     alpha=0.2)
    ax.grid(True)
    if i == 2:
        ax.set_xlabel('Time [seconds]')
    for i in range(number_of_orbits + 1):
        ax.axvline(i * period, color='gray', linestyle='--')
plt.show()
fig3.savefig(file_path + 'ErrMRP.png', dpi=200, bbox_inches='tight')


# Plot prefit and postfit samples for MRPs
print("Plotting prefit and postfit samples for MRPs")
plt.figure()
plt.plot(prefit_samples[0,:], alpha=0.5)
plt.plot(posfit_samples[0,:], alpha=0.5)
plt.grid('on')
plt.xlabel('Samples'); plt.ylabel('Star tracker residuals (unitless)')
plt.legend(["Prefit", "Postfit"])
plt.savefig(file_path + 'ResdStar.png', dpi=200, bbox_inches='tight')

# Plot prefit and postfit samples for omegas
print("Plotting prefit and postfit samples for omegas")
plt.figure()
plt.plot(prefit_samples[1,:], alpha=0.5)
plt.plot(posfit_samples[1,:], alpha=0.5)
plt.grid('on')
plt.xlabel('Samples'); plt.ylabel('Angular velocity meas residuals (rad/s)')
plt.legend(["Prefit", "Postfit"])
plt.savefig(file_path + 'ResdOmega.png', dpi=200, bbox_inches='tight')

# Plot histograms of the residuals
print("Plotting histograms of the residuals")
plt.figure()
plt.hist(prefit_samples[0,:], bins=50, alpha=0.5, density=True)
plt.hist(posfit_samples[0,:], bins=50, alpha=0.5, density=True)
plt.grid('on')
plt.xlabel('Star tracker residuals (unitless)');
plt.ylabel('Estimated PDF')
plt.legend(["Prefit", "Postfit"])
plt.savefig(file_path + 'HistStarRes.png', dpi=200, bbox_inches='tight')

plt.figure()
plt.hist(prefit_samples[1,:], bins=50, alpha=0.5, density=True)
plt.hist(posfit_samples[1,:], bins=50, alpha=0.5, density=True)
plt.grid('on')
plt.xlabel('Angular velocity meas residuals (rad/s)');
plt.ylabel('Estimated PDF')
plt.legend(["Prefit", "Postfit"])
plt.savefig(file_path + 'HistOmegaRes.png', dpi=200, bbox_inches='tight')