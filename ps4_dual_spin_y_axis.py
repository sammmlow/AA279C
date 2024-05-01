# -*- coding: utf-8 -*-

# Part 1: Assume that 2 components of the initial angular velocities are zero,
# and that the principal axes are aligned with the inertial frame (e.g., zero
# Euler angles). Verify that during the simulation the 2 components of angular
# velocity remain zero, and that the attitude represents a pure rotation about
# the rotation axis (e.g., linearly increasing Euler angle). Plot velocities
# and angles.

import numpy as np
import matplotlib.pyplot as plt
from source.spacecraft import Spacecraft
from source.attitudes import QTR, MRP
from numpy.linalg import norm

plt.close("all")

# Initial parameters.
geo_elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6];

# Each initial omega for three scenarios are columns of...
omega_magnitude = 3.0 # Degrees per second
initial_omegas = omega_magnitude * np.deg2rad( np.eye(3) );

# The principal inertia tensor of The Nimble Ladybug is...
initial_inertia = np.diag( [4770.398, 6313.894, 7413.202] );

# Initialize the wheel moment of inertia.
wheel_density = 4430 # Ti-6Al-4V [kg/m^3]
wheel_mass = wheel_density * np.pi * 0.4 * 0.4 * 0.1
wheel_inertia = 0.5 * wheel_mass * 0.4 * 0.4
wheel_omega = 300 * 2 * np.pi / 60.0

# Constant. Not going to change.
wheel_momentum = np.array([0, wheel_inertia * wheel_omega, 0])
wheel_energy = 0.5 * wheel_inertia * wheel_omega * wheel_omega

# The initial attitude is a quaternion aligned to inertial.
initial_attitude = QTR( dcm = np.identity(3) );

# Initialize simulation time parameters.
duration, timestep = 3600, 0.01
samples = int(duration / timestep) + 1
timeAxis = np.linspace(0, duration, samples)

# Three test cases: for each spacecraft, the initial angular
# velocity is set to 1 degree for one axis, with a randomized
# randomized perturbation added to it.
for axis in [0, 1, 2]:
    
    n = 0
    now = 0.0
    
    # Perturb the initial conditions,
    perturbation = 0.1 * omega_magnitude * np.deg2rad( np.random.rand(3) );
    perturbed_omega = initial_omegas[axis] + perturbation;
    
    # Initialize the spacecraft with perturbed omega
    sc = Spacecraft( elements = geo_elements,
                     ohmBN = perturbed_omega,
                     attBN = initial_attitude,
                     inertia = initial_inertia )
    
    # Setup containers of states.
    states_omega = np.zeros(( samples, 3 ))
    states_angle = np.zeros(( samples, 3 ))
    states_momentum = np.zeros(( samples, 3 ))
    states_energy = np.zeros(samples)
    
    while now <= duration:
        
        # Store the angular velocities and 321 Euler angles
        states_omega[n,:] = np.array([sc.ohmBN[0], sc.ohmBN[1], sc.ohmBN[2]])
        states_angle[n,:] = sc.attBN.get_euler_angles_321()
        
        # Compute the total momentum and energy, make sure it is conserved.
        # It must be emphasised that component-wise angular momentum is
        # conserved in the inertial frame. Need to multiply it by the DCM!
        Iw = sc.inertia @ sc.ohmBN
        states_momentum[n,:] = sc.attBN.dcm @ (Iw + wheel_momentum)
        states_energy[n] = 0.5 * np.dot(sc.ohmBN, Iw) + wheel_energy
        
        # Propagate the attitude and the angular velocity. Propagate attitude 
        # with a wheel takes as arguments: dt, torque, IR, ohmR, ohmRDot, vecR
        sc.propagate_orbit(timestep)
        sc.propagate_attitude_with_a_wheel(
            timestep, [0,0,0], wheel_inertia, wheel_omega, 0.0, [0,1,0])
        
        now += timestep
        n += 1
    
    # Plot Euler angles.
    fig1, axes1 = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
    labels = ['Roll \u03C6', 'Pitch \u03B8', 'Yaw \u03C8']  # psi, theta, phi
    for i, ax in enumerate(axes1):
        ax.plot( timeAxis, states_angle[:,i] * 57.3 )
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
        ax.plot( timeAxis, states_omega[:,i] )
        ax.set_ylabel(labels[i] + ' [rad/s]')
        ax.grid(True)
        if i == 2:
            ax.set_xlabel('Time [seconds]')
            
    # Plot energy and momentum.
    fig3, axes3 = plt.subplots(nrows=2, ncols=1, figsize=(7, 6))
    labels = ['Momentum [kg m^2 / s]', 'Energy [kg m^2 / s^2]']
    for i, ax in enumerate(axes3):
        if i == 0:
            ax.plot( timeAxis, states_momentum[:,0] )
            ax.plot( timeAxis, states_momentum[:,1] )
            ax.plot( timeAxis, states_momentum[:,2] )
            ax.legend(['$L_x$','$L_y$','$L_z$'])
            ax.set_ylabel(labels[i])
        if i == 1:
            ax.plot( timeAxis, states_energy, color='r' )
            ax.set_ylabel(labels[i])
            ax.set_xlabel('Time [seconds]')
        ax.grid(True)