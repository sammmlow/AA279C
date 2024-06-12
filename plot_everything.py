# -*- coding: utf-8 -*-

# Script for us to just plot everything. Can append new plots here,

import numpy as np
import matplotlib.pyplot as plt
from source.plot_orbit_and_attitude import plot_orbit_and_attitude

# Feel free to throw more inputs into this function as we plot more stuff!
# The 3D plotting significantly slows down the plotting code. Set parameter
# `plot_orbit_bool` to True if you need to plot 3D orbits.

empty = np.array([])

def plot_everything( timeAxis, skip, period, number_of_periods, file_path,  
                     states_quatr, states_gtorq, states_mtorq,
                     states_storq, states_angle, states_omega,
                     states_pos, states_pos_sampled, states_dcm_sampled,
                     states_quatr_ref = empty, states_omega_ref = empty,
                     plot_orbit_bool = False, states_ctrl = [],
                     plot_orbit_filename = ''):
    
    # ========================================================================
    # Plot body-to-inertial quaternions.
    # ========================================================================
    
    if (states_quatr.size != 0):
        
        print("Plotting BN quaternions: body-to-inertial")
        plt.figure()
        plt.plot( timeAxis[::skip], states_quatr[0,::skip] )
        plt.plot( timeAxis[::skip], states_quatr[1,::skip] )
        plt.plot( timeAxis[::skip], states_quatr[2,::skip] )
        plt.plot( timeAxis[::skip], states_quatr[3,::skip] )
        plt.xlabel('Simulation time [sec]')
        plt.ylabel('Quaternions (BN)')
        plt.legend(['q0','q1','q2','q3'])
    
        # Plot the orbital periods as vertical lines.
        for i in range(number_of_periods + 1):
            plt.axvline(i * period, color='gray', linestyle='--')
    
        plt.grid()
        plt.show()
        plt.savefig(file_path + 'QTR-BN.png', dpi=200, bbox_inches='tight')
        
    # ========================================================================
    # Plot body-to-reference quaternions.
    # ========================================================================
    
    if (states_quatr.size != 0):
        
        print("Plotting BR quaternions: body-to-reference")
        plt.figure()
        plt.plot( timeAxis[::skip], states_quatr_ref[0,::skip] )
        plt.plot( timeAxis[::skip], states_quatr_ref[1,::skip] )
        plt.plot( timeAxis[::skip], states_quatr_ref[2,::skip] )
        plt.plot( timeAxis[::skip], states_quatr_ref[3,::skip] )
        plt.xlabel('Simulation time [sec]')
        plt.ylabel('Quaternions (BR)')
        plt.legend(['q0','q1','q2','q3'])
    
        # Plot the orbital periods as vertical lines.
        for i in range(number_of_periods + 1):
            plt.axvline(i * period, color='gray', linestyle='--')
    
        plt.grid()
        plt.show()
        plt.savefig(file_path + 'QTR-BR.png', dpi=200, bbox_inches='tight')
    
    # ========================================================================
    # Plot gravity gradient torques
    # ========================================================================
    
    if (states_gtorq.size != 0):
        
        print("Plotting torque: gravity gradient")
        max_grav_grad = 21.1e-6 # N m
    
        plt.figure()
        plt.plot( timeAxis[::skip], states_gtorq[0,::skip] )
        plt.plot( timeAxis[::skip], states_gtorq[1,::skip] )
        plt.plot( timeAxis[::skip], states_gtorq[2,::skip] )
        plt.hlines( max_grav_grad, timeAxis[0], timeAxis[-1], colors='r')
        plt.hlines(-max_grav_grad, timeAxis[0], timeAxis[-1], colors='r')
        plt.xlabel('Simulation time [sec]')
        plt.ylabel('Gravity Gradient Torque in Principal-Body Axis [N m]')
        plt.legend(['$G_x$','$G_y$','$G_z$','Max'])
    
        # Plot the orbital periods as vertical lines.
        for i in range(number_of_periods + 1):
            plt.axvline(i * period, color='gray', linestyle='--')
    
        plt.grid()
        plt.show()
        plt.savefig(file_path + 'gTorque.png', dpi=200, bbox_inches='tight')
    
    # ========================================================================
    # Plot magnetic dipole moment torques
    # ========================================================================
    
    if (states_mtorq.size != 0):
        
        print("Plotting torque: magnetic dipole moments")
        max_mag_torque = 4.9e-6 # N m
    
        plt.figure()
        plt.plot( timeAxis[::skip], states_mtorq[0,::skip] )
        plt.plot( timeAxis[::skip], states_mtorq[1,::skip] )
        plt.plot( timeAxis[::skip], states_mtorq[2,::skip] )
        plt.hlines( max_mag_torque, timeAxis[0], timeAxis[-1], colors='r')
        plt.hlines(-max_mag_torque, timeAxis[0], timeAxis[-1], colors='r')
        plt.xlabel('Simulation time [sec]')
        plt.ylabel('Magnetic Moment Torque in Principal-Body Axis [N m]')
        plt.legend(['$B_x$','$B_y$','$B_z$','Max'])
    
        # Plot the orbital periods as vertical lines.
        for i in range(number_of_periods + 1):
            plt.axvline(i * period, color='gray', linestyle='--')
    
        plt.grid()
        plt.show()
        plt.savefig(file_path + 'mTorque.png', dpi=200, bbox_inches='tight')

    # ========================================================================
    # Plot solar radiation pressure torques
    # ========================================================================
    
    if (states_storq.size != 0):
        
        print("Plotting torque: solar radiation pressure")
        max_srp_torque = 96.6e-6 # N m
        
        plt.figure()
        plt.plot( timeAxis[::skip], states_storq[0,::skip] )
        plt.plot( timeAxis[::skip], states_storq[1,::skip] )
        plt.plot( timeAxis[::skip], states_storq[2,::skip] )
        plt.hlines( max_srp_torque, timeAxis[0], timeAxis[-1], colors='r')
        plt.hlines(-max_srp_torque, timeAxis[0], timeAxis[-1], colors='r')
        plt.xlabel('Simulation time [sec]')
        plt.ylabel('Solar Radiation Pressure Torque in Principal-Body Axis [N m]')
        plt.legend(['$S_x$','$S_y$','$S_z$','Max'])
    
        # Plot the orbital periods as vertical lines.
        for i in range(number_of_periods + 1):
            plt.axvline(i * period, color='gray', linestyle='--')
    
        plt.grid()
        plt.show()
        plt.savefig(file_path + 'sTorque-Pert.png', dpi=200, bbox_inches='tight')
    
    # ========================================================================
    # Plot Euler angles
    # ========================================================================
    
    if (states_angle.size != 0):
        
        print("Plotting 3-2-1 Euler angles (yaw-pitch-roll)")
        fig1, axes1 = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
        labels = ['Roll \u03C6', 'Pitch \u03B8', 'Yaw \u03C8']
        for i, ax in enumerate(axes1):
            ax.plot( timeAxis[::skip], states_angle[i,::skip] * 57.3 )
            ax.set_ylabel(labels[i] + ' [deg]')
            ax.set_ylim(-200, 200)
            ax.axhline(-180, color='gray', linestyle='--')
            ax.axhline( 180, color='gray', linestyle='--')
            ax.grid(True)
            if i == 2:
                ax.set_xlabel('Time [seconds]')
            for i in range(number_of_periods + 1):
                ax.axvline(i * period, color='gray', linestyle='--')
        plt.show()
        plt.savefig(file_path + 'Angles.png', dpi=200, bbox_inches='tight')
    
    # ========================================================================
    # Plot body-to-inertial angular velocities in body-frame coordinates
    # ========================================================================
    
    if (states_omega.size != 0):
        
        print("Plotting BN angular velocities (body-frame)")
        fig2a, axes2a = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
        labels = [r'$\omega_{x}$', r'$\omega_{y}$', r'$\omega_{z}$']
        for i, ax in enumerate(axes2a):
            ax.plot( timeAxis[::skip], states_omega[i,::skip] )
            ax.set_ylabel(labels[i] + '(BN) [rad/s]')
            ax.grid(True)
            if i == 2:
                ax.set_xlabel('Time [seconds]')
            for i in range(number_of_periods + 1):
                ax.axvline(i * period, color='gray', linestyle='--')
        plt.show()
        plt.savefig(file_path + 'OmegaBN.png', dpi=200, bbox_inches='tight')
        
    # ========================================================================
    # Plot body-to-reference angular velocities in body-frame coordinates
    # ========================================================================
    
    if (states_omega_ref.size != 0):
        
        print("Plotting BR angular velocities (body-frame)")
        fig2b, axes2b = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
        labels = [r'$\omega_{x}$', r'$\omega_{y}$', r'$\omega_{z}$']
        for i, ax in enumerate(axes2b):
            ax.plot( timeAxis[::skip], states_omega_ref[i,::skip] )
            ax.set_ylabel(labels[i] + '(BR) [rad/s]')
            ax.grid(True)
            if i == 2:
                ax.set_xlabel('Time [seconds]')
            for i in range(number_of_periods + 1):
                ax.axvline(i * period, color='gray', linestyle='--')
        plt.show()
        plt.savefig(file_path + 'OmegaBR.png', dpi=200, bbox_inches='tight')
    
    # ========================================================================
    # Final: Plot orbit and attitude triads in 3D (slow!)
    # ========================================================================
    
    if (plot_orbit_bool == True and states_pos_sampled.size != 0):
    
        print("Plotting orbit and attitude triads (3D)")
        figOrbit = plt.figure(figsize=(10, 10))
        axesOrbit = figOrbit.add_subplot(111, projection='3d')
        plot_orbit_and_attitude(axesOrbit,
                                states_pos[0, ::skip],
                                states_pos[1, ::skip],
                                states_pos[2, ::skip], 
                                states_pos_sampled, 
                                states_dcm_sampled)
        plt.tight_layout()
        strFile = plot_orbit_filename + '-Orbit3D.png'
        plt.savefig(file_path + strFile, dpi=200, bbox_inches='tight')
        
    # ========================================================================
    # Plot controller torques
    # ========================================================================
    
    if (states_ctrl.size != 0):
        
        print("Plotting torque: controller torque")
    
        plt.figure()
        plt.plot( timeAxis[::skip], states_ctrl[0,::skip] )
        plt.plot( timeAxis[::skip], states_ctrl[1,::skip] )
        plt.plot( timeAxis[::skip], states_ctrl[2,::skip] )
        plt.xlabel('Simulation time [sec]')
        plt.ylabel('Controller Torque in Principal-Body Axis [N m]')
        plt.legend(['$M_x$','$M_y$','$M_z$'])
    
        # Plot the orbital periods as vertical lines.
        for i in range(number_of_periods + 1):
            plt.axvline(i * period, color='gray', linestyle='--')
    
        plt.grid()
        plt.show()
        plt.savefig(file_path + 'CtrlTorque.png', dpi=200, bbox_inches='tight')