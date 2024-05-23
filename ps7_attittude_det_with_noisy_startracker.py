# -*- coding: utf-8 -*-
# ===========================================================================
# In this scenario, we assess attitude determination with stars.
# ===========================================================================

import datetime
import numpy as np
import matplotlib.pyplot as plt

from source import perturbations
from source import ephemeris
from source.spacecraft import Spacecraft
from source.attitudes import QTR #, MRP
import source.attitude_estimation as att_est
import source.unit_vec_gaussian as uvg

# from plot_everything import plot_everything

# For saving the figures
# file_path = "figures/ps7/PS7-AttDet-StarTracker-Statistical" 
# file_path = "figures/ps7/PS7-AttDet-StarTracker-Deterministic"
# file_path = "figures/ps7/PS7-AttDet-Gyro"
file_path = "figures/ps7/PS7-AttDet-SunMag-Statistical"
current_time = datetime.datetime(2025, 1, 1, 12, 0, 0) # For ECEF computation.
plt.close("all")

# ===========================================================================
# Toggle these flags to determine the fate of this script.
# ===========================================================================

bool_enable_perturbations = True
bool_enable_active_control = True
bool_plot_orbit = True
bool_use_statistical_estimation = True
bool_ang_vel_mismatch = False

# Statistical
# Deterministic

# ===========================================================================
# Set the negative feedback gains of a simple proportional controller.
# ===========================================================================

K_attBR = [-5E-5, -5E-5, -2E-3]
K_ohmBR = [-2E-3, -2E-3, -4E-3]

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
# Set up the attitude estimator...
# ===========================================================================

if bool_use_statistical_estimation is not None:
    if bool_use_statistical_estimation:
        estimator = att_est.StatisticalAttitudeEstimator(verbose=False)
    else:
        estimator = att_est.DeterministicAttitudeEstimator(use_projection=True)

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
# Import Stars...
# ===========================================================================

num_stars = 10
star_catalog = f"hipparcos_star_catalog_solutions_downselected_{num_stars}.txt"
stars = np.loadtxt(star_catalog, delimiter = " ")[:, :3].T

assert stars.shape[0] == 3, "Number of rows is not 3!" + \
    f" Expected 3, got {stars.shape[1]}."
assert stars.shape[1] == num_stars, "Number of stars do not match!" + \
    f" Expected {num_stars}, got {stars.shape[0]}."


# ===========================================================================
# Star Tracker Noise...
# ===========================================================================

star_tracker_noise_uvg = uvg.UnitVecGaussian(
    mean_angle = np.deg2rad(0.0025),
    std_dev_angle = np.deg2rad(0.01)
)

# ===========================================================================
# Gyro Noise...
# ===========================================================================

delta_t_hrs = timestep / 3600 # hours
gyro_drift_per_hr = 0.0018 # degrees per hour
gyro_drift = gyro_drift_per_hr * delta_t_hrs
print("Gyro drift: ", gyro_drift)

gyro_noise_uvg = uvg.UnitVecGaussian(
    mean_angle = np.deg2rad(0.2),
    std_dev_angle = np.deg2rad(gyro_drift)
)

# ===========================================================================
# Sun Sensor and Magnetometer Noise...
# ===========================================================================

sun_sensor_noise_uvg = uvg.UnitVecGaussian(
    mean_angle = np.deg2rad(0.5),
    std_dev_angle = np.deg2rad(2.5)
)

magnetometer_noise_uvg = uvg.UnitVecGaussian(
    mean_angle = np.deg2rad(1),
    std_dev_angle = np.deg2rad(2.5)
)

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

states_pos_sampled = np.zeros(( 3, samples ))
states_dcm_sampled = np.zeros(( 3, 3, samples ))

att_est_qtr_att = np.zeros(( 4, samples ))
att_est_total_err = np.zeros(samples)
att_est_qtr_att_err = np.zeros(( 4, samples ))


# ===========================================================================
# Attitude estimation with angular velocity only...
# ===========================================================================

sc_angvel = Spacecraft( elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6],
                 inertia = np.diag( [4770.398, 6313.894, 7413.202] ) )

# Body (B) to inertial (N) angular velocity and attitude
sc_angvel.ohmBN = -initial_omega
sc_angvel.attBN = QTR( dcm = initial_dcm )

att_est_ang_vel_only = np.zeros(( 3, samples ))
att_est_ang_vel_only_qtr = np.zeros(( 4, samples ))
att_est_ang_vel_only_err = np.zeros(( 4, samples ))

att_est_ang_vel_only[:, 0] = sc_angvel.ohmBN
att_est_ang_vel_only_qtr[:, 0] = sc_angvel.attBN

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

    # Compute controller torque using state feedback.
    if bool_enable_active_control:
        ctrl_gyro_terms = 2 * np.cross(ohmBR, sc.inertia) @ sc.ohmBN
        ctrl_torque = K_attBR * qtrBR[1:] + K_ohmBR * ohmBR + ctrl_gyro_terms

    # Propagate the attitude and the angular velocity
    sc.propagate_orbit(timestep)
    sc.propagate_attitude(timestep, torque = pert_torque + ctrl_torque )

    # Repeat the same for the angular velocity only estimator.
    sc_angvel.propagate_orbit(timestep)

    if bool_ang_vel_mismatch:
        # tot_torque = 0.5 * pert_torque + ctrl_torque
        # Sample a drift angle for the gyro.
        # Certainly, this is not the best way to do this.
        _, drift_angle, _ = gyro_noise_uvg.sample(n_samples=1, 
                                                  return_axis_angle=True)
        
        # sc_avg_inertia = np.mean(np.diag(sc.inertia))
        torque_sample = drift_angle.flatten() / (timestep ** 3)
        # print("Perturbation Torque: ", pert_torque)
        # print("Control Torque: ", ctrl_torque)
        # print("Sampled Torque: ", torque_sample)

        tot_torque = pert_torque + ctrl_torque + torque_sample
    else:
        tot_torque = pert_torque + ctrl_torque

    sc_angvel.propagate_attitude(timestep, torque = tot_torque )

    att_est_ang_vel_only[:, n] = sc_angvel.ohmBN
    ang_vel_only_gt_rot = sc.attBN.dcm.T
    ang_vel_only_est_rot = sc_angvel.attBN.dcm.T

    ang_vel_only_att_err_dcm = ang_vel_only_gt_rot @ ang_vel_only_est_rot.T
    ang_vel_only_att_err_qtr = QTR(dcm = ang_vel_only_att_err_dcm)

    att_est_ang_vel_only_qtr[:, n] = sc_angvel.attBN.qtr
    att_est_ang_vel_only_err[:, n] = ang_vel_only_att_err_qtr.qtr

    # SUN SENSOR AND MAGNETOMETER
    if bool_use_statistical_estimation is not None:

        # Get the direction vector to the sun.
        sun_pos_eci = ephemeris.compute_sun_position_eci( current_time )
        sun_direction_eci = sun_pos_eci / np.linalg.norm(sun_pos_eci)
        sun_direction_body = sc.attBN.dcm.T @ sun_direction_eci

        # Get the magnetic field vector.
        mag_vec_body, mag_vec_eci = ephemeris.compute_magnetic_direction( 
            current_time, sc.states[0:3], sc.attBN )
        
        # Combine the sun and magnetic field vectors.
        sun_mag_vec_body = np.concatenate((sun_direction_body, mag_vec_body))
        sun_mag_vec_eci = np.concatenate((sun_direction_eci, mag_vec_eci))

        #########################


        # Get the star measurements.
        gt_rot = sc.attBN.dcm.T # i.e., from ECI to body frame.
        star_measurements = gt_rot @ stars

        # Add noise to the star measurements.
        samples = star_tracker_noise_uvg.sample(num_stars)
        star_meas_noisy = star_measurements.copy()
        # Apply the rotation to the samples.
        for i in range(num_stars):
            qtr_curr = QTR(qtr = samples[i, :])
            qtr_curr.conventionalize()
            star_meas_noisy[:, i] = qtr_curr.apply(star_meas_noisy[:, i])

        # Attitude determination with stars.
        dcm_att_estimate = estimator.estimate(star_meas_noisy, stars)
        qtr_att_estimate = QTR(dcm = dcm_att_estimate)
        att_est_qtr_att[:, n] = qtr_att_estimate.qtr

        # print("Attitude estimate: \n", dcm_att_estimate)
        # print("GT Attitude: \n", gt_rot)

        # Compute the measurement error.
        _, tot_err = estimator.verify_estimate(star_meas_noisy, stars, 
                                               dcm_att_estimate)
        # print("Total error: ", tot_err)
        att_est_total_err[n] = tot_err

        # Compute the attitude error.
        # att_err = qtr_att_estimate / sc.attBN
        # print("Attitude error: ", att_err)
        # att_est_qtr_att_err[:, n] = att_err.qtr
        att_err_dcm = gt_rot @ dcm_att_estimate.T
        att_err_qtr = QTR(dcm = att_err_dcm)

        # qtr_att_estimate_conjugate = [ qtr_att_estimate.qtr[0],
        #                               -qtr_att_estimate.qtr[1],
        #                               -qtr_att_estimate.qtr[2],
        #                               -qtr_att_estimate.qtr[3]]
        # print(qtr_att_estimate_conjugate)
        # att_err_qtr_conj = QTR(qtr = qtr_att_estimate_conjugate)

        # attBN_conjugate = [ sc.attBN.qtr[0],
        #                    -sc.attBN.qtr[1],
        #                    -sc.attBN.qtr[2],
        #                    -sc.attBN.qtr[3]]
        # attBN_qtr_conj = QTR(qtr = attBN_conjugate)
        # att_err_qtr_direct = sc.attBN * att_err_qtr #QTR(qtr = qtr_att_estimate_conjugate)

        # print("Attitude error (from DCM): ", att_err_qtr.qtr)
        # print("Attitude error (from QTR): ", att_err_qtr_direct.qtr)

        att_est_qtr_att_err[:, n] = att_err_qtr.qtr


    # STAR TRACKER
    if bool_use_statistical_estimation is not None:

        # Get the star measurements.
        gt_rot = sc.attBN.dcm.T # i.e., from ECI to body frame.
        star_measurements = gt_rot @ stars

        # Add noise to the star measurements.
        samples = star_tracker_noise_uvg.sample(num_stars)
        star_meas_noisy = star_measurements.copy()
        # Apply the rotation to the samples.
        for i in range(num_stars):
            qtr_curr = QTR(qtr = samples[i, :])
            qtr_curr.conventionalize()
            star_meas_noisy[:, i] = qtr_curr.apply(star_meas_noisy[:, i])

        # Attitude determination with stars.
        dcm_att_estimate = estimator.estimate(star_meas_noisy, stars)
        qtr_att_estimate = QTR(dcm = dcm_att_estimate)
        att_est_qtr_att[:, n] = qtr_att_estimate.qtr

        # print("Attitude estimate: \n", dcm_att_estimate)
        # print("GT Attitude: \n", gt_rot)

        # Compute the measurement error.
        _, tot_err = estimator.verify_estimate(star_meas_noisy, stars, 
                                               dcm_att_estimate)
        # print("Total error: ", tot_err)
        att_est_total_err[n] = tot_err

        # Compute the attitude error.
        # att_err = qtr_att_estimate / sc.attBN
        # print("Attitude error: ", att_err)
        # att_est_qtr_att_err[:, n] = att_err.qtr
        att_err_dcm = gt_rot @ dcm_att_estimate.T
        att_err_qtr = QTR(dcm = att_err_dcm)

        # qtr_att_estimate_conjugate = [ qtr_att_estimate.qtr[0],
        #                               -qtr_att_estimate.qtr[1],
        #                               -qtr_att_estimate.qtr[2],
        #                               -qtr_att_estimate.qtr[3]]
        # print(qtr_att_estimate_conjugate)
        # att_err_qtr_conj = QTR(qtr = qtr_att_estimate_conjugate)

        # attBN_conjugate = [ sc.attBN.qtr[0],
        #                    -sc.attBN.qtr[1],
        #                    -sc.attBN.qtr[2],
        #                    -sc.attBN.qtr[3]]
        # attBN_qtr_conj = QTR(qtr = attBN_conjugate)
        # att_err_qtr_direct = sc.attBN * att_err_qtr #QTR(qtr = qtr_att_estimate_conjugate)

        # print("Attitude error (from DCM): ", att_err_qtr.qtr)
        # print("Attitude error (from QTR): ", att_err_qtr_direct.qtr)

        att_est_qtr_att_err[:, n] = att_err_qtr.qtr

    # Update simulation time and calendar time
    current_time = current_time + datetime.timedelta(seconds = timestep)
    now += timestep
    n += 1

# ===========================================================================
# Plot everything!
# ===========================================================================

if bool_use_statistical_estimation is not None:
    print("Plotting Total Measurement Error")
    fig = plt.figure()
    plt.plot(timeAxis[::skip], att_est_total_err[::skip], label = "Total Error")
    plt.xlabel('Simulation time [sec]')
    plt.ylabel('Total L2 Measurement Error')

    # Plot the orbital periods as vertical lines.
    for i in range(number_of_orbits + 1):
        plt.axvline(i * period, color='gray', linestyle='--')

    plt.grid()
    plt.show()
    fig.savefig(f"{file_path}-TotalError.png")

    print("Plotting Quaternion Attitude Error")
    fig = plt.figure()
    plt.plot(timeAxis[::skip], att_est_qtr_att_err[0, ::skip])
    plt.plot(timeAxis[::skip], att_est_qtr_att_err[1, ::skip])
    plt.plot(timeAxis[::skip], att_est_qtr_att_err[2, ::skip])
    plt.plot(timeAxis[::skip], att_est_qtr_att_err[3, ::skip])
    plt.xlabel('Simulation time [sec]')
    plt.ylabel('Error Quaternions (BR vs BR estimate)')
    plt.legend(['q0','q1','q2','q3'])

    # Plot the orbital periods as vertical lines.
    for i in range(number_of_orbits + 1):
        plt.axvline(i * period, color='gray', linestyle='--')

    plt.grid()
    plt.show()
    fig.savefig(file_path + 'QTR-BR.png', dpi=200, bbox_inches='tight')


    print("Plotting (vector part) Quaternion Attitude Error")
    fig = plt.figure()
    # plt.plot(timeAxis[::skip], att_est_qtr_att_err[0, ::skip])
    plt.plot(timeAxis[::skip], att_est_qtr_att_err[1, ::skip])
    plt.plot(timeAxis[::skip], att_est_qtr_att_err[2, ::skip])
    plt.plot(timeAxis[::skip], att_est_qtr_att_err[3, ::skip])
    plt.xlabel('Simulation time [sec]')
    plt.ylabel('Error Quaternions (BR vs BR estimate)')
    # plt.legend(['q0','q1','q2','q3'])
    plt.legend(['q1','q2','q3'])

    # Plot the orbital periods as vertical lines.
    for i in range(number_of_orbits + 1):
        plt.axvline(i * period, color='gray', linestyle='--')

    plt.grid()
    plt.show()
    fig.savefig(file_path + 'QTR-BR-three.png', dpi=200, bbox_inches='tight')


else:
    print("Plotting Quaternion Attitude Error for Angular Velocity Only Estimation")
    fig = plt.figure()
    plt.plot(timeAxis[::skip], att_est_ang_vel_only_err[0, ::skip])
    plt.plot(timeAxis[::skip], att_est_ang_vel_only_err[1, ::skip])
    plt.plot(timeAxis[::skip], att_est_ang_vel_only_err[2, ::skip])
    plt.plot(timeAxis[::skip], att_est_ang_vel_only_err[3, ::skip])
    plt.xlabel('Simulation time [sec]')
    plt.ylabel('Error Quaternions (BR vs BR estimate)')
    plt.legend(['q0','q1','q2','q3'])

    # Plot the orbital periods as vertical lines.
    for i in range(number_of_orbits + 1):
        plt.axvline(i * period, color='gray', linestyle='--')

    plt.grid()
    plt.show()
    fig.savefig(file_path + 'QTR-BR.png', dpi=200, bbox_inches='tight')


    print("Plotting (vector part) Quaternion Attitude Error for Angular Velocity Only Estimation")
    fig = plt.figure()
    # plt.plot(timeAxis[::skip], att_est_ang_vel_only_err[0, ::skip])
    plt.plot(timeAxis[::skip], att_est_ang_vel_only_err[1, ::skip])
    plt.plot(timeAxis[::skip], att_est_ang_vel_only_err[2, ::skip])
    plt.plot(timeAxis[::skip], att_est_ang_vel_only_err[3, ::skip])
    plt.xlabel('Simulation time [sec]')
    plt.ylabel('Error Quaternions (BR vs BR estimate)')
    # plt.legend(['q0','q1','q2','q3'])
    plt.legend(['q1','q2','q3'])

    # Plot the orbital periods as vertical lines.
    for i in range(number_of_orbits + 1):
        plt.axvline(i * period, color='gray', linestyle='--')

    plt.grid()
    plt.show()
    fig.savefig(file_path + 'QTR-BR-three.png', dpi=200, bbox_inches='tight')


# plot_everything( timeAxis, skip, period, number_of_orbits, file_path,
#                   states_qtrBN, states_gtorq, states_mtorq,
#                   states_storq, states_angle, states_ohmBN,
#                   states_pos, states_pos_sampled, states_dcm_sampled,
#                   states_qtrBR, states_ohmBR, bool_plot_orbit )