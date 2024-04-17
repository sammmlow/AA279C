# -*- coding: utf-8 -*-

import numpy as np
from source.spacecraft import Spacecraft

# Define the default osculating orbital elements [km and degrees].
[a, e, i, w, R, M] = [6878.137, 0.001, 97.5976, 0.0, 240.0, 0.827]

# Initialize four spacecraft.
sc1 = Spacecraft(elements = [a,e,i,w,R,M])
sc2 = Spacecraft(elements = [a,e,i,w,R,M])
sc3 = Spacecraft(elements = [a,e,i,w,R,M])
sc4 = Spacecraft(elements = [a,e,i,w,R,M])

sc3.forces['j2'] = True   # On a numerical propagator
sc4.forces['j2'] = True   # On a numerical propagator
sc4.forces['drag'] = True # On a numerical propagator

# Prepare three container matrices for comparison of states
now, duration, timestep, n = 0.0, 864000, 60.0, 0
samples = int(duration / timestep)
states = np.zeros(( samples, 3, 4 )) # Samples x Coords x Spacecraft

# Run a loop and propagate all three spacecraft.
while now < duration:
    
    # Record the states.
    states[ n, 0:3, 0 ] = np.array([sc1.px, sc1.py, sc1.pz])
    states[ n, 0:3, 1 ] = np.array([sc2.px, sc2.py, sc2.pz])
    states[ n, 0:3, 2 ] = np.array([sc3.px, sc3.py, sc3.pz])
    states[ n, 0:3, 3 ] = np.array([sc4.px, sc4.py, sc4.pz])
    
    # Propagate the spacecraft
    sc1.propagate_orbit( timestep )
    sc2.propagate_perturbed( timestep, timestep )
    sc3.propagate_perturbed( timestep, timestep )
    sc4.propagate_perturbed( timestep, timestep )
    
    # Update time and sample count.
    now += timestep
    n += 1

# Plot the results.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_orbit(axis, x, y, z):
    radius = 6378.140
    phi, theta = np.mgrid[0:np.pi:100j, 0:2*np.pi:100j]
    sx = radius * np.sin(phi)* np.cos(theta)
    sy = radius  *np.sin(phi)* np.sin(theta)
    sz = radius * np.cos(phi)
    axis.plot_surface(sx, sy, sz, color='b', alpha=0.2)
    axis.plot(x, y, z)
    axis.set_xlabel('X [km]')
    axis.set_ylabel('Y [km]')
    axis.set_zlabel('Z [km]')

fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_subplot(221, projection='3d')
plot_orbit(ax1, states[:,0,0], states[:,1,0], states[:,2,0])

ax2 = fig.add_subplot(222, projection='3d')
plot_orbit(ax2, states[:,0,1], states[:,1,1], states[:,2,1])

ax3 = fig.add_subplot(223, projection='3d')
plot_orbit(ax3, states[:,0,2], states[:,1,2], states[:,2,2])

ax4 = fig.add_subplot(224, projection='3d')
plot_orbit(ax4, states[:,0,3], states[:,1,3], states[:,2,3])
