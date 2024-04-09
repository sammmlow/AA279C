# -*- coding: utf-8 -*-

###############################################################################
###############################################################################
##                                                                           ##
##    AA279C Example Code with the Python Spacecraft Class                   ##
##    Created on Mon Apr  8 15:50:45 2024                                    ##
##                                                                           ##
##    Since this is an ADCS course, this spacecraft class abstracts away     ##
##    all needs for translational motion (e.g. orbit propagation), so that   ##
##    we can really focus on the ADCS part of the class. There are only      ##
##    rudimentary implementations of attitude state representations, for     ##
##    example quaternions and Rodrigues parameters. Attitude propagation     ##
##    is also implemented in this class using Euler's rotational equations   ##
##    of motion. This is only for convenience, so please do not feel         ##
##    obliged to use this class unless you want to!                          ##
##                                                                           ##
###############################################################################
###############################################################################

from source import spacecraft

# Example 1: Define the spacecraft, its orbital elements, and give it a name!

orb_a = 6878.137 # 500 km altitude
orb_e = 0.001    # degrees
orb_i = 97.5976  # degrees
orb_w = 0.0      # degrees
orb_R = 250.662  # degrees
orb_M = 0.827    # degrees

elemList = [orb_a, orb_e, orb_i, orb_w, orb_R, orb_M]

sc = spacecraft.Spacecraft(elements = elemList)

sc.name = 'Boaty McBoat-Face'

sc.status() # This prints out the state of the S/C at current time.

###############################################################################
###############################################################################

# Example 2: We can perform orbit propagation on the spacecraft either 
# analytically using Kepler's Laws, or numerically using an RK4 propagator.

import numpy as np

timeNow = 0.0
duration = 86400 * 1
timestep = 30.0
samples = int(duration / timestep)
n = 0; # Sample count

# Propagate the different spacecraft and compare.
sc1 = spacecraft.Spacecraft(elements = elemList, name='Boaty Analytical')
sc2 = spacecraft.Spacecraft(elements = elemList, name='Boaty RK4, No J2')
sc3 = spacecraft.Spacecraft(elements = elemList, name='Boaty RK4, Yes J2')
sc4 = spacecraft.Spacecraft(elements = elemList, name='Boaty RK4, Drag')

# Toggle the force models for each spacecraft. The current implemented force
# models are "twobody" (default), "j2", "drag", and "maneuvers" (RTN frame).

sc1.forces['j2'] = False  # On an analytical propagator
sc2.forces['j2'] = False
sc3.forces['j2'] = True
sc4.forces['drag'] = True

# Prepare three container matrices for comparison of states

states_sc1 = np.zeros((samples,3)) # Keplerian
states_sc2 = np.zeros((samples,3)) # RK4 without J2
states_sc3 = np.zeros((samples,3)) # RK4 with J2
states_sc4 = np.zeros((samples,3)) # Drag

# Run a loop and propagate all three spacecraft.

while timeNow < duration:
    
    # Record the states.
    
    states_sc1[n,0:3] = np.array([sc1.px, sc1.py, sc1.pz])
    states_sc2[n,0:3] = np.array([sc2.px, sc2.py, sc2.pz])
    states_sc3[n,0:3] = np.array([sc3.px, sc3.py, sc3.pz])
    states_sc4[n,0:3] = np.array([sc4.px, sc4.py, sc4.pz])
    
    # Propagate the spacecrafts
    
    sc1.propagate_orbit( timestep )
    sc2.propagate_perturbed( timestep, timestep )
    sc3.propagate_perturbed( timestep, timestep )
    sc4.propagate_perturbed( timestep, timestep )
    
    # Update time and sample count.
    
    timeNow += timestep
    n += 1

# Plot the results.

import matplotlib.pyplot as plt

plt.close("all")

fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(states_sc1[:,0], states_sc1[:,1], states_sc1[:,2])

ax2 = fig.add_subplot(222, projection='3d')
ax2.plot(states_sc2[:,0], states_sc2[:,1], states_sc2[:,2])

ax3 = fig.add_subplot(223, projection='3d')
ax3.plot(states_sc3[:,0], states_sc3[:,1], states_sc3[:,2])

ax4 = fig.add_subplot(224, projection='3d')
ax4.plot(states_sc4[:,0], states_sc4[:,1], states_sc4[:,2])

plt.show()

###############################################################################
###############################################################################

# Example 3: We can also assign an initial quaternion to it, add a non-zero
# angular velocity, and propagate Boaty McBoat-Face under torque-free motion
# (I haven't validated the attitude segment of the code before though, the
# attitude library has been work in progress and I never really finished it).

# Print out the default initialized quaternion (body-to-inertial) of Boaty
print("Initial S/C quaternion = ", sc.attBN, "\n")

# The above is a Quaternion type (defined in attitudes.py). To use the 
# quaternion as a NumPy array, you can access it by the .qtr attribute,
print("Initial S/C quaternion (as NumPy array) = ", sc.attBN.qtr, "\n")

# You can also print the attitude as a direction cosine,
print("Initial S/C quaternion (as a DCM) = ")
print(sc.attBN.dcm, "\n")

# Convert a rotation matrix into a quaternion
from source.rotation import dcmX, dcmY, dcmZ
arbitrary_rotation = dcmX(1) @ dcmY(1) @ dcmZ(1)

# Initialize a quaternion with this arbitrary rotation,
from source.attitudes import QTR
arbitrary_quaternion = QTR( dcm = arbitrary_rotation )
print("Arbitrary quaternion = ", arbitrary_quaternion, "\n")

# Perform quaternion multiplication on current body to inertial attitude
# (see attitudes.py for implementation)
sc.attBN = sc.attBN * arbitrary_quaternion

# Print out its corresponding quaternion
print("Final S/C quaternion = ", sc.attBN, "\n")

# Print out the resulting DCM
print("Final S/C quaternion (as a DCM) = ")
print(sc.attBN.dcm)

# Note that an application of quaternion operations on the spacecraft attitude
# will effect changes to both the DCM and quaternions. States are consistent.
# However, applying a rotation to the spacecraft attitude's DCM does not 
# update the quaternion. Thus, the state consistency is only one-way, we
# should avoid attitude updates via DCMs, and work with quaternions directly.

# There are also other attitude implementations, but I haven't tested those
# against a third party software (unlike the orbit propagation one)...

# Hope these tools help make our work more efficient for this course!