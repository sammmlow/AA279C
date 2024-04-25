# -*- coding: utf-8 -*-

import numpy as np
from source.spacecraft import Spacecraft
from source.attitudes import QTR, MRP

# Define the default osculating orbital elements [km and degrees].
initial_elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6]
initial_omega = [0.0873, 0.0698, 0.0524]
initial_inertia = np.diag([4770.398, 6313.894, 7413.202])

zero_torque = np.array([0.0, 0.0, 0.0])

# Set duration, time step, and a container of angular velocity states
now, duration, timestep, n = 0.0, 360, 1, 0
samples = int(duration / timestep)

# Define container variables for the angular velocities and the attitudes
states_omega = np.zeros(( samples, 3, 2 )) # samples x coords x types
states_attitude = np.zeros(( samples, 4, 2 )) # samples x coords x types

# Initialize the Ladybug using Quaternions.
sc1 = Spacecraft(elements = initial_elements,
                ohmBN = initial_omega,
                inertia = initial_inertia)

sc1.attBN = QTR([1, 0, 0, 0])  # Set initial body to inertial attitude
sc1.attBR = QTR([1, 0, 0, 0])  # Set initial body to reference attitude

# Re-initialize the Ladybug using MRPs.
sc2 = Spacecraft(elements = initial_elements,
                ohmBN = initial_omega,
                inertia = initial_inertia)

sc2.attBN = MRP([0, 0, 0])  # Set initial body to inertial attitude
sc2.attBR = MRP([0, 0, 0])  # Set initial body to reference attitude

while now < duration:
    
    states_omega[n, 0:3, 0] = np.array(
        [ sc1.ohmBN[0], sc1.ohmBN[1], sc1.ohmBN[2] ])
    states_omega[n, 0:3, 1] = np.array(
        [ sc2.ohmBN[0], sc2.ohmBN[1], sc2.ohmBN[2] ])
    
    
    
    states_attitude[n, 0:4, 0] = np.array(
        [ sc1.attBN[0], sc1.attBN[1], sc1.attBN[2], sc1.attBN[3] ])  # QTR
    states_attitude[n, 0:3, 1] = np.array(
        [ sc2.attBN[0], sc2.attBN[1], sc2.attBN[2] ])                # MRP
    
    sc1.propagate_attitude(timestep, zero_torque)
    sc2.propagate_attitude(timestep, zero_torque)
    
    now += timestep
    n += 1
    
import matplotlib.pyplot as plt

plt.close("all")
timeAxis = np.linspace(0,duration,samples)

def plot_omega(time, x, y, z):
    plt.figure()
    plt.plot( time, x )
    plt.plot( time, y )
    plt.plot( time, z )
    plt.xlabel('Simulation time [sec]')
    plt.ylabel('Body-Inertial Angular Velocity (Body Coordinates) [rad/s]')
    plt.legend(['Omega X [rad/s]', 'Omega Y [rad/s]', 'Omega Z [rad/s]'])
    plt.grid()
    plt.show()

def plot_attBN_qtr(time, q0, q1, q2, q3):
    plt.figure()
    plt.plot( time, q0 )
    plt.plot( time, q1 )
    plt.plot( time, q2 )
    plt.plot( time, q3 )
    plt.xlabel('Simulation time [sec]')
    plt.ylabel('Body-to-Inertial Quaternions')
    plt.legend(['q0','q1','q2','q3'])
    plt.grid()
    plt.show()

def plot_attBN_mrp(time, m1, m2, m3):
    plt.figure()
    plt.plot( time, m1 )
    plt.plot( time, m2 )
    plt.plot( time, m3 )
    plt.xlabel('Simulation time [sec]')
    plt.ylabel('Body-to-Inertial Modified Rodrigues Parameters')
    plt.legend(['m1','m2','m3'])
    plt.grid()
    plt.show()

plot_omega(timeAxis,
           states_omega[:,0,0],
           states_omega[:,1,0],
           states_omega[:,2,0])

plot_attBN_qtr(timeAxis,
               states_attitude[:,0,0],
               states_attitude[:,1,0],
               states_attitude[:,2,0],
               states_attitude[:,3,0])

plot_omega(timeAxis,
           states_omega[:,0,1],
           states_omega[:,1,1],
           states_omega[:,2,1])

plot_attBN_mrp(timeAxis,
               states_attitude[:,0,1],
               states_attitude[:,1,1],
               states_attitude[:,2,1])

# Print the final DCM and check they are close?
print(sc1.attBN.dcm)
print(sc2.attBN.dcm)


