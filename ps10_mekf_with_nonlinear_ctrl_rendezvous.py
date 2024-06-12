# -*- coding: utf-8 -*-

import datetime
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

from source import perturbations
from source.spacecraft import Spacecraft
from source.attitudes import QTR, MRP
from source.rotation import dcmX, dcmY, dcmZ

np.random.seed(seed=1)

file_path = "figures/ps10/PS10-FullADCS-"

# Copy over the star catalog into a matrix.
nStars = 10 # Hardcoded.
catalog = np.genfromtxt(
    "hipparcos_star_catalog_solutions_downselected_10.txt", delimiter=' ')

current_time = datetime.datetime(2025, 1, 1, 12, 0, 0) # For ECEF computation.
plt.close("all")

# ===========================================================================
# Toggle these flags to determine the fate of this script.
# ===========================================================================

bool_enable_perturbations = True
bool_enable_active_control = True
bool_plot_orbit = True

# ===========================================================================
# Parameters for non-linear MRP-based Lyapunov control law
# ===========================================================================

Kp = -np.array([5E-8, 1E-6, 5E-7])
Kd = -np.array([1E-4, 1E-4, 1E-5])

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
    updated_mean = STM @ mean
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
        H_mrp[ (3 * n):(3 * n + 3), 0:3 ] = 4 * DCM_N2B @ skew(catalog[n, 0:3])
    ZN3 = np.zeros((3 * N, 3))
    H = np.block([[H_mrp, ZN3, ZN3], [Z3, I3, I3]])
    
    # Compute prefit residual L2-norms,
    prefit_stars = norm(meas_stars - model_stars, 2) / nStars
    prefit_omega = norm(meas_omega - model_omega, 2)
    prefit = np.array([ prefit_stars, prefit_omega ])
    
    # Full residual vector. Transpose before reshape so that the residuals per
    # unit direction are all contiguous.
    residuals = np.concatenate([ (meas_stars - model_stars).T.reshape(3 * N),
                                 (meas_omega - model_omega) ])
    # Residuals are unit vectors expressed in body frame 
    # Kalman gain, state mean and cov update.
    I9 = np.identity(9)
    K = cov @ H.T @ np.linalg.inv(H @ cov @ H.T + R)
    IKH = I9 - K @ H
    KRK = K @ R @ K.T
    updated_mean = mean + K @ residuals
    updated_cov = IKH @ cov @ IKH.T + KRK
    
    # Update the quaternion in the spacecraft model by correcting with dMRP.
    error_mrps = MRP( mrp = updated_mean[0:3] )
    updated_quat = QTR( dcm = error_mrps.dcm.T @ quat.dcm )
    
    # Generate postfit measurements now
    DCM_N2B_POST = updated_quat.dcm.T
    postfit_model_stars = DCM_N2B_POST @ catalog[:, 0:3].T
    postfit_model_omega = updated_mean[3:6] + updated_mean[6:9]
    postfit = np.array([ norm(meas_stars - postfit_model_stars, 2) / nStars,
                         norm(meas_omega - postfit_model_omega, 2) ])
    
    return [updated_mean, updated_cov, updated_quat, prefit, postfit]


# ===========================================================================
# Functions to generate noisy measurements. Use ground truth to generate them.
# ===========================================================================

noise_rad = 0.000175 # 0.01 degrees
noise_omega = 0.0000175 # 0.001 degrees/s

def make_noisy_dcm():
    x = dcmX( np.random.normal(0.0, noise_rad) )
    y = dcmY( np.random.normal(0.0, noise_rad) )
    z = dcmZ( np.random.normal(0.0, noise_rad) )
    return x @ y @ z

def make_noisy_measurements( catalog, true_quaternion, true_omega, bias ):
    k = 0
    meas_omega = np.zeros(3)
    meas_stars = np.zeros( np.shape(catalog) )
    for star in catalog:
        DCM_N2B = true_quaternion.dcm.T
        meas_stars[k, 0:3] = DCM_B2S @ DCM_N2B @ make_noisy_dcm() @ star[:3]
        k += 1
    add_noise_omega = np.random.normal(0.0, noise_omega)
    meas_omega = (make_noisy_dcm() @ true_omega) + add_noise_omega + bias
    return [meas_stars, meas_omega]


# ===========================================================================
# Set simulation time and sampling parameters...
# ===========================================================================

period = 86400
number_of_orbits = 2
duration = number_of_orbits * period
now, n, timestep = 0.0, 0, 120.0
samples = int(duration / timestep)

skip = 2 # Larger skip leads to faster plotting

timeAxis = np.linspace(0, duration, samples)

print("Number of steps: ", samples)
print("Sample skip rate: ", skip)
print("Number of plotted samples: ", samples // skip)


# ===========================================================================
# Containers for storing states for plotting later on...
# ===========================================================================

# For state estimator

errors_mrp  = np.zeros(( 3, samples )) # Part of the state
errors_ohm  = np.zeros(( 3, samples )) # Part of the state
errors_bias = np.zeros(( 3, samples )) # Part of the state
errors_qtr  = np.zeros(( 4, samples )) # NOT part of the state

prefit_samples = np.zeros(( 2, samples ))
posfit_samples = np.zeros(( 2, samples ))

covariance_history = np.zeros(( 9, 9, samples + 1 ))  # Full history

# For ground truth and controller

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

# ===========================================================================
# Containers for plotting DCM triads on the orbit plots...
# ===========================================================================
 
big_samples = 12

sample_idx_1 = 0 # Just a counter for plotting attitude triads in 3D...
sample_time_trigger_1 = np.linspace(0, period, big_samples + 1)[:-1]
states_pos_sampled_1 = np.zeros(( 3, big_samples ))    # Pre-rendezvous
states_dcm_sampled_1 = np.zeros(( 3, 3, big_samples )) # Pre-rendezvous

sample_idx_2 = 0 # Just a counter for plotting attitude triads in 3D...
sample_time_trigger_2 = np.linspace(period, 2*period, big_samples + 1)[:-1]
states_pos_sampled_2 = np.zeros(( 3, big_samples ))    # During rendezvous
states_dcm_sampled_2 = np.zeros(( 3, 3, big_samples )) # During rendezvous


# ===========================================================================
# Initialize ground truth parameters and true spacecraft in RTN configuration.
# ===========================================================================

sc = Spacecraft( elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6],
                 inertia = np.diag([4770.398, 6313.894, 7413.202]) )

initial_omega = -np.array([0, 0, sc.n])
initial_dcm = sc.get_hill_frame().T  # RTN2ECI

slew_for_rendezvous = dcmZ(-np.pi/2) # For rotation mid-operation.

sc.ohmBN = initial_omega
sc.attBN = QTR( dcm = initial_dcm )

true_bias = 0.0001 * np.ones(3) # rad/s


# ===========================================================================
# Initialize the filter states, covariances, and parameters. The filter has 
# been (very painstakingly) fine-tuned manually! Exercise caution below!
# ===========================================================================

quaternion = QTR( dcm = make_noisy_dcm() @ initial_dcm )
mean = np.zeros(9)

# Initialize with some noisy omega plus bias.
mean[3:6] = make_noisy_dcm() @ initial_omega

# Initial covariance: [Error MRPs, omegas, and omega biases]
covariance = np.diag([5E-5] * 3 + [5E-9] * 3 + [5E-7] * 3)
covariance_history[:, :, 0] = covariance

# Manually fine-tuned filter process noise
Q = np.diag([1E-5] * 3 + [1E-12] * 3 + [1E-14] * 3)

# Manually fine-tuned filter measurement noise
R = np.diag([1E-5] * (3 * nStars) + [5E-10] * 3)


# ===========================================================================
# Actual dynamics simulation, with the MEKF operations called below.
# ===========================================================================

while now < duration:
    
    # Store spacecraft states.
    states_pos[:, n] = sc.states[0:3]
    states_ohmBN[:, n] = sc.ohmBN
    states_qtrBN[:, n] = sc.attBN.qtr
    states_angle[:, n] = sc.attBN.get_euler_angles_321()

    # Compute reference-to-inertial omegas and attitudes.
    
    # First stage of the rendezvous process is simply RTN aligned.
    # The following set of hardcoded gains work well...
    if (now <= period):
        Kp = -np.array([5E-8, 1E-6, 5E-7])
        Kd = -np.array([1E-4, 1E-4, 1E-5])
        ohmRN = initial_omega
        attRN = QTR( dcm = sc.get_hill_frame().T ) # RTN2ECI
        
        # Sample DCM for plotting triads later on.
        if sample_idx_1 < big_samples:
            if (now >= sample_time_trigger_1[ sample_idx_1 ]):
                states_pos_sampled_1[:, sample_idx_1] = sc.states[0:3]
                states_dcm_sampled_1[:, :, sample_idx_1] = sc.attBN.dcm
                sample_idx_1 += 1
        
    # Second stage requires negative X to be pointing in along-track.
    else:
        print("Getting ready for rendezvous!")
        Kp = -np.array([1E-5, 2E-7, 3E-6])
        Kd = -np.array([4E-3, 1E-4, 1.5E-4])
        ohmRN = slew_for_rendezvous @ initial_omega
        attRN = QTR( dcm = slew_for_rendezvous @ sc.get_hill_frame().T )
        
        # Sample DCM for plotting triads later on.
        if sample_idx_2 < big_samples:
            if (now >= sample_time_trigger_2[ sample_idx_2 ]):
                states_pos_sampled_2[:, sample_idx_2] = sc.states[0:3]
                states_dcm_sampled_2[:, :, sample_idx_2] = sc.attBN.dcm
                sample_idx_2 += 1
    
    # Compute body-to-reference (controller error) omegas and attitudes.
    # Note that we are NOT using sc.attBN, which is ground truth. Rather,
    # we are applying `quaternion` instead, which is estimated.
    est_omega = mean[3:6]
    est_attitude = quaternion
    ohmBR = est_omega - ohmRN    # Feedback into controller
    qtrBR = est_attitude / attRN # Feedback into controller

    # Represent as short-only rotation for qtrBR.
    # Very important for the controller!
    if qtrBR[0] < 0.0:
        qtrBR.qtr = -1 * qtrBR.qtr
    
    print(ohmBR)
    print(qtrBR.qtr)
    print("")

    # Store body-to-reference (controller error) omegas and attitudes.
    states_ohmBR[:, n] = ohmBR
    states_qtrBR[:, n] = qtrBR
    
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
        ctrl_torque = Kp * (skew(mrpBR.mrp) @ sc.inertia @ mrpBR.mrp)
        ctrl_torque += Kd * (sc.inertia @ ohmBR)
        # ctrl_torque = np.zeros(3)
    
    # Store control torque.
    states_ctrl[:, n] = ctrl_torque
    
    # Propagate the true spacecraft attitude and the angular velocity
    sc.propagate_orbit(timestep)
    sc.propagate_attitude(timestep, torque = pert_torque + ctrl_torque )
    
    # Perform the MEKF time update
    mean, covariance, quaternion = time_update(
        timestep, mean, covariance, quaternion, ctrl_torque, Q )
    
    # Generate noisy measurements of stars and angular velocities.
    meas_stars, meas_omega = make_noisy_measurements(
        catalog, sc.attBN, sc.ohmBN, true_bias )
    
    # Perform the MEKF measurement update
    mean, covariance, updated_quaternion, prefit, postfit = meas_update(
        mean, covariance, quaternion, catalog, meas_stars, meas_omega, R )
    
    # Save prefit and postfit samples
    prefit_samples[:, n] = prefit
    posfit_samples[:, n] = postfit
    
    # Compute the quaternion error
    quaternion_error = updated_quaternion / sc.attBN
    quaternion_error.conventionalize()
    
    # Record state errors for plotting later on.
    errors_mrp[:, n]  = mean[0:3]
    errors_ohm[:, n]  = mean[3:6] - sc.ohmBN
    errors_bias[:, n] = mean[6:9] - true_bias
    errors_qtr[:, n]  = quaternion_error
    
    # Record covariance in covariance history.
    covariance_history[:, :, n] = covariance
    quaternion = updated_quaternion
    
    # Reset the error MRPs to zero.
    mean = np.concatenate([ [0,0,0], mean[3:9] ])
    
    # Update simulation time and calendar time
    current_time = current_time + datetime.timedelta(seconds = timestep)
    now += timestep
    n += 1


# ===========================================================================
# Plot state estimates
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
    for i in range(int(number_of_orbits) + 1):
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
    ax.set_ylim([-0.0003, 0.0003])
    ax.grid(True)
    if i == 2:
        ax.set_xlabel('Time [seconds]')
    for i in range(int(number_of_orbits) + 1):
        ax.axvline(i * period, color='gray', linestyle='--')
plt.show()
fig1b.savefig(file_path + 'OmegaBiases.png', dpi=200, bbox_inches='tight')


# Plotting error of the updated reference quaternions
print("Plotting error of the updated reference quaternions")
fig2, axes2 = plt.subplots(nrows=4, ncols=1, figsize=(7, 6))
labels = ['$\Delta q_0 (BN)$',
          '$\Delta q_1 (BN)$',
          '$\Delta q_2 (BN)$',
          '$\Delta q_3 (BN)$']
for i, ax in enumerate(axes2):
    ax.plot( timeAxis[::skip], errors_qtr[i,::skip])
    ax.set_ylabel(labels[i])
    ax.set_ylim([-1.1, 1.1])
    ax.grid(True)
    if i == 3:
        ax.set_xlabel('Time [seconds]')
    for i in range(int(number_of_orbits) + 1):
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
    for i in range(int(number_of_orbits) + 1):
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

# # Plot histograms of the residuals
# print("Plotting histograms of the residuals")
# plt.figure()
# plt.hist(prefit_samples[0,:], bins=50, alpha=0.5, density=True)
# plt.hist(posfit_samples[0,:], bins=50, alpha=0.5, density=True)
# plt.grid('on')
# plt.xlabel('Star tracker residuals (unitless)');
# plt.ylabel('Estimated PDF')
# plt.legend(["Prefit", "Postfit"])
# plt.savefig(file_path + 'HistStarRes.png', dpi=200, bbox_inches='tight')

# plt.figure()
# plt.hist(prefit_samples[1,:], bins=50, alpha=0.5, density=True)
# plt.hist(posfit_samples[1,:], bins=50, alpha=0.5, density=True)
# plt.grid('on')
# plt.xlabel('Angular velocity meas residuals (rad/s)');
# plt.ylabel('Estimated PDF')
# plt.legend(["Prefit", "Postfit"])
# plt.savefig(file_path + 'HistOmegaRes.png', dpi=200, bbox_inches='tight')

# # ===========================================================================
# # Plot everything!
# # ===========================================================================

from plot_everything import plot_everything

nil = np.array([])

plot_everything( timeAxis, skip, period, number_of_orbits, file_path,
                  states_qtrBN, nil, nil, nil, states_angle, states_ohmBN,
                  states_pos, nil, nil, states_qtrBR, states_ohmBR,
                  False, states_ctrl )

# Just plot pre-rendezvous orbit plots...
plot_everything( timeAxis, skip, period, number_of_orbits, file_path,
                  nil, nil, nil, nil, nil, nil, states_pos, 
                  states_pos_sampled_1, states_dcm_sampled_1,
                  nil, nil, bool_plot_orbit, nil, 'Period1-')

# Just plot during-rendezvous orbit plots...
plot_everything( timeAxis, skip, period, number_of_orbits, file_path,
                  nil, nil, nil, nil, nil, nil, states_pos, 
                  states_pos_sampled_2, states_dcm_sampled_2,
                  nil, nil, bool_plot_orbit, nil, 'Period2-')

# Add plot mapping control torques to individual reaction wheel torques.
A_rw_inv = (np.sqrt(3)/8.0) * np.array([[ 1,  1,  1],
                                        [ 1,  1, -1],
                                        [ 1, -1,  1],
                                        [ 1, -1, -1],
                                        [-1,  1,  1],
                                        [-1,  1, -1],
                                        [-1, -1,  1],
                                        [-1, -1, -1]])
max_rw_torque = 0.25
rw_torques = A_rw_inv @ states_ctrl

plt.figure()
for i in range(8):
    plt.plot( timeAxis[::skip], rw_torques[i, ::skip], alpha = 0.75 )
plt.grid()
plt.show()
plt.xlabel('Simulation time [sec]')
plt.ylabel('Individual Reaction Wheel Torques [N m]')
plt.legend(['RW1','RW2','RW3','RW4','RW5','RW6','RW7','RW8'])
plt.savefig(file_path + 'RW-Torques.png', dpi=200, bbox_inches='tight')