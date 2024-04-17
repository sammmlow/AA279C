# -*- coding: utf-8 -*-

import numpy as np
from source.spacecraft import Spacecraft

# Define the default osculating orbital elements [km and degrees].
initial_elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6]
initial_omega = [0.0873, 0.0698, 0.0524]
initial_inertia = np.diag([4770.398, 6313.894, 7413.202])

# Initialize the Ladybug, with a non-zero angular velocity.
sc = Spacecraft(elements = initial_elements,
                ohmBN = initial_omega,
                inertia = initial_inertia)

# Set duration, time step, and a container of angular velocity states
now, duration, timestep, n = 0.0, 360, 1.0, 0
samples = int(duration / timestep)
states = np.zeros(( samples, 3 ))

# Run a loop and propagate all three spacecraft.
while now < duration:
    states[n,:] = np.array([ sc.ohmBN[0], sc.ohmBN[1], sc.ohmBN[2] ])
    zero_torque = np.array([0.0, 0.0, 0.0])
    sc.propagate_attitude(timestep, zero_torque)
    now += timestep
    n += 1

# Plot the results.

import matplotlib.pyplot as plt

def plot_omega(time, x, y, z):
    plt.figure()
    plt.plot( timeAxis, x )
    plt.plot( timeAxis, y )
    plt.plot( timeAxis, z )
    plt.xlabel('Simulation time [sec]')
    plt.ylabel('Body-Inertial Angular Velocity (Body Coordinates) [rad/s]')
    plt.legend(['Omega X [rad/s]', 'Omega Y [rad/s]', 'Omega Z [rad/s]'])
    plt.grid()
    plt.show()

plt.close("all")
timeAxis = np.linspace(0,duration,samples)

plot_omega(timeAxis, states[:,0], states[:,1], states[:,2])
