# -*- coding: utf-8 -*-

# Part 1: Assume that 2 components of the initial angular velocities are zero,
# and that the principal axes are aligned with the inertial frame (e.g., zero
# Euler angles). Verify that during the simulation the 2 components of angular
# velocity remain zero, and that the attitude represents a pure rotation about
# the rotation axis (e.g., linearly increasing Euler angle). Plot velocities
# and angles.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D
from source.spacecraft import Spacecraft
from source.attitudes import QTR, MRP
from source.plot_orbit_and_attitude import plot_orbit_and_attitude

## ===========================================================================
## ACTUAL CODE BEGINS HERE!
## ===========================================================================

plt.close("all")

# Initialize a spacecraft with these GEO elements.
geo_elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6];
sc = Spacecraft( elements = geo_elements )

# Initialize the angular velocity as the mean motion
initial_omega = [0, 0, sc.n];

# The principal inertia tensor of The Nimble Ladybug is...
initial_inertia = np.diag( [4770.398, 6313.894, 7413.202] );

# The initial attitude is a quaternion aligned to the SC's RTN.
initial_attitude = QTR( dcm = sc.get_hill_frame().T );

# Set the initial omegas, attitudes, and inertias.
sc.ohmBN = initial_omega
sc.attBN = initial_attitude

# Initialize simulation time parameters.
now, n, duration, timestep = 0.0, 0, 86400, 60.0
samples = int(duration / timestep)
timeAxis = np.linspace(0, duration, samples)
sample_bigstep = 4
sample_trigger = duration / sample_bigstep # Fragile code. 

# Initialize containers for plotting.
x = np.zeros(samples)
y = np.zeros(samples)
z = np.zeros(samples)
xyz_sampled = np.zeros(( 3, samples ))
dcm_sampled = np.zeros(( 3, 3, samples ))
states_omega = np.zeros(( 3, samples ))
states_angle = np.zeros(( 3, samples ))
states_quatr = np.zeros(( 4, samples ))

nBig = 0

# Propagate attitude and orbit
while now < duration:
    
    # Store the angular velocities and 321 Euler angles
    x[n] = sc.states[0];
    y[n] = sc.states[1];
    z[n] = sc.states[2];
    states_omega[:, n] = sc.ohmBN
    states_angle[:, n] = sc.attBN.get_euler_angles_321()
    states_quatr[:, n] = sc.attBN.qtr
    
    # Fragile code. Will not work if time step skips this.
    if (now % sample_trigger == 0):
        xyz_sampled[:, nBig] = sc.states[0:3]
        dcm_sampled[:, :, nBig] = sc.attBN.dcm
        nBig += 1
    
    # Propagate the attitude and the angular velocity
    sc.propagate_orbit(timestep)
    sc.propagate_attitude(timestep, torque=[0,0,0])
    
    now += timestep
    n += 1
    
# Plot quaternions.
plt.figure()
plt.plot( timeAxis, states_quatr[0,:] )
plt.plot( timeAxis, states_quatr[1,:] )
plt.plot( timeAxis, states_quatr[2,:] )
plt.plot( timeAxis, states_quatr[3,:] )
plt.xlabel('Simulation time [sec]')
plt.ylabel('Body-to-Inertial Quaternions')
plt.legend(['q0','q1','q2','q3'])
plt.grid()
plt.show()
    
# Plot Euler angles.
fig1, axes1 = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
labels = ['Roll \u03C6', 'Pitch \u03B8', 'Yaw \u03C8']  # psi, theta, phi
for i, ax in enumerate(axes1):
    ax.plot( timeAxis, states_angle[i,:] * 57.3 )
    ax.set_ylabel(labels[i] + ' [deg]')
    ax.set_ylim(-200, 200)
    ax.axhline(-180, color='gray', linestyle='--')
    ax.axhline( 180, color='gray', linestyle='--')
    ax.grid(True)
    if i == 2:
        ax.set_xlabel('Time [seconds]')
    
# Plot angular velocities.
fig2, axes2 = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
labels = [r'$\omega_{x}$', r'$\omega_{y}$', r'$\omega_{z}$']
for i, ax in enumerate(axes2):
    ax.plot( timeAxis, states_omega[i,:] )
    ax.set_ylabel(labels[i] + ' [rad/s]')
    ax.grid(True)
    if i == 2:
        ax.set_xlabel('Time [seconds]')
        
# Plot visualization of RTN orbit
fig3 = plt.figure(figsize=(10, 10))
axes3 = fig3.add_subplot(111, projection='3d')
plot_orbit_and_attitude(axes3, x, y, z, xyz_sampled, dcm_sampled)