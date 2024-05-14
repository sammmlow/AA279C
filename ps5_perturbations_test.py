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
x = np.zeros(samples)
y = np.zeros(samples)
z = np.zeros(samples)
xyz_sampled = np.zeros(( 3, samples ))
dcm_sampled = np.zeros(( 3, 3, samples ))
states_omega = np.zeros(( 3, samples ))
states_angle = np.zeros(( 3, samples ))
states_quatr = np.zeros(( 4, samples ))
states_gtorq = np.zeros(( 3, samples ))
states_mtorq = np.zeros(( 3, samples ))
states_storq = np.zeros(( 3, samples ))

# Just a counter for plotting the number of attitude triads in the 3D plot.
nBig = 0

# Make this number bigger to plot faster with fewer points.
sampleSkip = 5
# sampleSkip = 40
print(f"with skip of {sampleSkip}, number of samples: ", 
      samples // sampleSkip)

sample_trigger_count = 0.0;

# Propagate attitude and orbit
while now < duration:
    
    # Store the angular velocities and 321 Euler angles
    x[n] = sc.states[0]
    y[n] = sc.states[1]
    z[n] = sc.states[2]
    states_omega[:, n] = sc.ohmBN
    states_angle[:, n] = sc.attBN.get_euler_angles_321()
    states_quatr[:, n] = sc.attBN.qtr
    
    # Fragile code. Will not work if time step skips this.
    if (now >= sample_trigger_count):
        xyz_sampled[:, nBig] = sc.states[0:3]
        dcm_sampled[:, :, nBig] = sc.attBN.dcm
        sample_trigger_count += sample_trigger_interval
        nBig += 1
        
    # Compute gravity gradient torque
    Rc_eci = np.array([x[n], y[n], z[n]])
    Rc_principal_body = sc.attBN.dcm.T @ Rc_eci
    
    gTorque = perturbations.compute_gravity_gradient_torque(
        sc.GM, Rc_principal_body, sc.inertia)
    
    mTorque = perturbations.compute_magnetic_torque_component(
        current_time, Rc_eci, sc.attBN)
    
    sTorque = perturbations.compute_solar_torque_component(
        current_time, Rc_eci, sc.attBN)
    
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
    
# Plot quaternions.
plt.figure()
plt.plot( timeAxis[::sampleSkip], states_quatr[0,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_quatr[1,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_quatr[2,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_quatr[3,::sampleSkip] )
plt.xlabel('Simulation time [sec]')
plt.ylabel('Body-to-Inertial Quaternions')
plt.legend(['q0','q1','q2','q3'])

# Plot the orbital periods as vertical lines.
for i in range(n_periods + 1):
    plt.axvline(i * one_orbital_period, color='gray', linestyle='--')

plt.grid()
plt.show()

# Save the quaternion plot
plt.savefig(file_path + 'QTR-Pert.png', dpi=200, bbox_inches='tight')

print("Plotting torque: gravity gradient")

max_grav_grad = 21.1e-6 # N m

# Plot gravity gradients.
plt.figure()
plt.plot( timeAxis[::sampleSkip], states_gtorq[0,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_gtorq[1,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_gtorq[2,::sampleSkip] )
plt.hlines( max_grav_grad, timeAxis[0], timeAxis[-1], colors='r')
plt.hlines(-max_grav_grad, timeAxis[0], timeAxis[-1], colors='r')
plt.xlabel('Simulation time [sec]')
plt.ylabel('Gravity Gradient Torque in Principal-Body Axis [N m]')
plt.legend(['$G_x$','$G_y$','$G_z$','Max'])

# Plot the orbital periods as vertical lines.
for i in range(n_periods + 1):
    plt.axvline(i * one_orbital_period, color='gray', linestyle='--')

plt.grid()
plt.show()

# Save the gravity gradient plot
plt.savefig(file_path + 'gTorque-Pert.png', dpi=200, bbox_inches='tight')

print("Plotting torque: magnetic")

max_mag_torque = 4.9e-6 # N m

# Plot magnetic moment torques.
plt.figure()
plt.plot( timeAxis[::sampleSkip], states_mtorq[0,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_mtorq[1,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_mtorq[2,::sampleSkip] )
plt.hlines( max_mag_torque, timeAxis[0], timeAxis[-1], colors='r')
plt.hlines(-max_mag_torque, timeAxis[0], timeAxis[-1], colors='r')
plt.xlabel('Simulation time [sec]')
plt.ylabel('Magnetic Moment Torque in Principal-Body Axis [N m]')
plt.legend(['$M_x$','$M_y$','$M_z$','Max'])

# Plot the orbital periods as vertical lines.
for i in range(n_periods + 1):
    plt.axvline(i * one_orbital_period, color='gray', linestyle='--')

plt.grid()
plt.show()

# Save the magnetic moment plot
plt.savefig(file_path + 'mTorque-Pert.png', dpi=200, bbox_inches='tight')

print("Plotting torque: SRP")

max_srp_torque = 96.6e-6 # N m

# Plot SRP moment torques.
plt.figure()
plt.plot( timeAxis[::sampleSkip], states_storq[0,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_storq[1,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_storq[2,::sampleSkip] )
plt.hlines( max_srp_torque, timeAxis[0], timeAxis[-1], colors='r')
plt.hlines(-max_srp_torque, timeAxis[0], timeAxis[-1], colors='r')
plt.xlabel('Simulation time [sec]')
plt.ylabel('Solar Radiation Pressure Torque in Principal-Body Axis [N m]')
plt.legend(['$S_x$','$S_y$','$S_z$','Max'])

# Plot the orbital periods as vertical lines.
for i in range(n_periods + 1):
    plt.axvline(i * one_orbital_period, color='gray', linestyle='--')

plt.grid()
plt.show()

# Save the SRP plot
plt.savefig(file_path + 'sTorque-Pert.png', dpi=200, bbox_inches='tight')
    
print("Plotting Euler")

# Plot Euler angles.
fig1, axes1 = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
labels = ['Roll \u03C6', 'Pitch \u03B8', 'Yaw \u03C8']  # psi, theta, phi
for i, ax in enumerate(axes1):
    ax.plot( timeAxis[::sampleSkip], states_angle[i,::sampleSkip] * 57.3 )
    ax.set_ylabel(labels[i] + ' [deg]')
    ax.set_ylim(-200, 200)
    ax.axhline(-180, color='gray', linestyle='--')
    ax.axhline( 180, color='gray', linestyle='--')
    ax.grid(True)
    if i == 2:
        ax.set_xlabel('Time [seconds]')
    
    # Plot the orbital periods as vertical lines.
    for i in range(n_periods + 1):
        ax.axvline(i * one_orbital_period, color='gray', linestyle='--')

plt.show()

# Save the Euler angle plot
plt.savefig(file_path + 'Angles-Pert.png', dpi=200, bbox_inches='tight')

print("Plotting angular velocities")

# Plot angular velocities.
fig2, axes2 = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
labels = [r'$\omega_{x}$', r'$\omega_{y}$', r'$\omega_{z}$']
for i, ax in enumerate(axes2):
    ax.plot( timeAxis[::sampleSkip], states_omega[i,::sampleSkip] )
    ax.set_ylabel(labels[i] + ' [rad/s]')
    ax.grid(True)
    if i == 2:
        ax.set_xlabel('Time [seconds]')
        
    # Plot the orbital periods as vertical lines.
    for i in range(n_periods + 1):
        ax.axvline(i * one_orbital_period, color='gray', linestyle='--')

# Save the angular velocity plot
plt.savefig(file_path + 'Omegas-Pert.png', dpi=200, bbox_inches='tight')

plt.show()

print("Plotting RTN")
# Plot visualization of RTN orbit
fig3 = plt.figure(figsize=(10, 10))
axes3 = fig3.add_subplot(111, projection='3d')
plot_orbit_and_attitude(axes3,
                        x[::sampleSkip], 
                        y[::sampleSkip], 
                        z[::sampleSkip], 
                        xyz_sampled, 
                        dcm_sampled)

plt.tight_layout()

# Save the RTN plot
plt.savefig(file_path + 'Orbit-Pert.png', dpi=200, bbox_inches='tight')