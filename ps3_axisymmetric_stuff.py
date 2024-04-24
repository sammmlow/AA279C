# -*- coding: utf-8 -*-

import numpy as np
from source.spacecraft import Spacecraft
from source.attitudes import QTR, MRP

# Define the default osculating orbital elements [km and degrees].
initial_elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6]
initial_omega = [0.0873, 0.0698, 0.0524]
initial_inertia = np.diag([4770.398, 4770.398, 7413.202])

zero_torque = np.array([0.0, 0.0, 0.0])

# Set duration, time step, and a container of angular velocity states
now, duration, timestep, n = 0.0, 360, 0.1, 0
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
    states_omega[n, 0:3, 0] = np.array(
        [ sc2.ohmBN[0], sc2.ohmBN[1], sc2.ohmBN[2] ])
    
    states_attitude[n, 0:4, 0] = np.array(
        [ sc1.attBN[0], sc1.attBN[1], sc1.attBN[2], sc1.attBN[3] ])  # QTR
    states_attitude[n, 0:3, 0] = np.array(
        [ sc2.attBN[0], sc2.attBN[1], sc2.attBN[2] ])                # MRP
    
    sc1.propagate_attitude(timestep, zero_torque)
    sc2.propagate_attitude(timestep, zero_torque)
    
    now += timestep
    n += 1


# Print the final DCM and check they are close?
print(sc1.attBN.dcm)
print(sc2.attBN.dcm)


