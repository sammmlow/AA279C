# -*- coding: utf-8 -*-

import numpy as np
from source.spacecraft import Spacecraft

# Define the default osculating orbital elements [km and degrees].
[a, e, i, w, R, M] = [6678.137, 0.001, 97.5976, 0.0, 240.0, 0.827]

# Initialize four spacecraft.
sc1 = Spacecraft(elements = [a,e,i,w,R,M])
sc2 = Spacecraft(elements = [a,e,i,w,R,M])
sc3 = Spacecraft(elements = [a,e,i,w,R,M])
sc4 = Spacecraft(elements = [a,e,i,w,R,M])

sc3.forces['j2'] = True   # On a numerical propagator
sc4.forces['j2'] = True   # On a numerical propagator
sc4.forces['drag'] = True # On a numerical propagator

# Prepare three container matrices for comparison of states
now, duration, timestep, n = 0.0, 864000, 10.0, 0
samples = int(duration / timestep)
states = np.zeros(( samples, 3, 4 )) # Samples x Coords x Spacecraft

# Run a loop and propagate all three spacecraft.
while now < duration:
    
    # Get the hill frame of the chief (SC1) to record the RTN states
    hill = sc1.get_hill_frame()
    
    sc1_eci = np.array([sc1.px, sc1.py, sc1.pz])
    sc2_eci = np.array([sc2.px, sc2.py, sc2.pz])
    sc3_eci = np.array([sc3.px, sc3.py, sc3.pz])
    sc4_eci = np.array([sc4.px, sc4.py, sc4.pz])
    
    sc2_rtn_wrt_sc1 = hill @ (sc2_eci - sc1_eci)
    sc3_rtn_wrt_sc1 = hill @ (sc3_eci - sc1_eci)
    sc4_rtn_wrt_sc1 = hill @ (sc4_eci - sc1_eci)
    
    # Record the states.
    states[ n, 0:3, 1 ] = sc2_rtn_wrt_sc1
    states[ n, 0:3, 2 ] = sc3_rtn_wrt_sc1
    states[ n, 0:3, 3 ] = sc4_rtn_wrt_sc1
    
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

def plot_rtn(time, r, t, n):
    plt.figure()
    plt.plot( timeAxis, r )
    plt.plot( timeAxis, t )
    plt.plot( timeAxis, n )
    plt.legend(['R [km]', 'T [km]', 'N [km]'])
    plt.xlabel('Simulation time [sec]')
    plt.ylabel('Propagation Differences in RTN [km]')
    plt.grid()
    plt.show()

plt.close("all")
timeAxis = np.linspace(0,duration,samples)

plot_rtn(timeAxis, states[:,0,1], states[:,1,1], states[:,2,1])
plot_rtn(timeAxis, states[:,0,2], states[:,1,2], states[:,2,2])
plot_rtn(timeAxis, states[:,0,3], states[:,1,3], states[:,2,3])
