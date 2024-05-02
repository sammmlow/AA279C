# -*- coding: utf-8 -*-

# Part 1: Assume that 2 components of the initial angular velocities are zero,
# and that the principal axes are aligned with the inertial frame (e.g., zero
# Euler angles). Verify that during the simulation the 2 components of angular
# velocity remain zero, and that the attitude represents a pure rotation about
# the rotation axis (e.g., linearly increasing Euler angle). Plot velocities
# and angles.

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D
from source.spacecraft import Spacecraft
from source.attitudes import QTR, MRP
from source.rotation import dcmX, dcmZ

# Compute gravity gradient torque. Rc in the principal body frame.
def compute_gravity_gradient_torque(GM, Rc, inertia):
    RcNorm = norm(Rc)
    k = 3 * GM / (RcNorm**3) # Note that km units cancel out here.
    Rcx, Rcy, Rcz = Rc / RcNorm
    Ix, Iy, Iz = np.diag(inertia)
    Mx = Rcy * Rcz * (Iz - Iy) * k
    My = Rcx * Rcz * (Ix - Iz) * k
    Mz = Rcx * Rcy * (Iy - Ix) * k
    return np.array([Mx, My, Mz])

# Define a function here that plots a satellite in orbit, with current
# attitude expressed as columns of its DCM (for visualization). 
# Parameters x, y, z, are arrays containing translational motion.
# xyz_sampled is a 3xN matrix comprising positions tagged to N samples.
# dcm_sampled is a 3x3xN tensor comprising DCMs tagged to N samples.
def plot_orbit_and_attitude(axis, x, y, z, xyz_sampled, dcm_sampled):
    
    # Sanity check that number of samples of positions match DCMs.
    N = np.shape(xyz_sampled)[1]
    if N != np.shape(dcm_sampled)[2]:
        raise ValueError("Number of pose samples do not match DCM samples!")
    
    # Rescale earth texture
    earth_texture = imread('earth.jpg')
    earth_reduced = earth_texture[::2,::2]
    earth_normalized = earth_reduced / 256 # rescale RGB values to [0,1]
    
    radius = 6378.140
    to_radians = np.pi/180
    lons = np.linspace(-180, 180, earth_reduced.shape[1]) * to_radians
    lats = np.linspace(-90, 90, earth_reduced.shape[0])[::-1] * to_radians
    
    sx = radius * np.outer(np.cos(lons), np.cos(lats)).T
    sy = radius * np.outer(np.sin(lons), np.cos(lats)).T
    sz = radius * np.outer(np.ones(np.size(lons)), np.sin(lats)).T
    axis.plot_surface(sx, sy, sz, 
                      facecolors = earth_normalized,
                      shade = False, alpha = 0.75, edgecolor = 'none')
    
    axis.plot(x, y, z)
    axis.set_xlabel('X [km]')
    axis.set_ylabel('Y [km]')
    axis.set_zlabel('Z [km]')
    axis.set_aspect('equal')
    
    # Plot individual attitudes.
    for n in range(N):
        xs = xyz_sampled[0,n]
        ys = xyz_sampled[1,n]
        zs = xyz_sampled[2,n]
        colors = ['b', 'g', 'r']
        for m in range(3):
            d1s = dcm_sampled[0,m,n]
            d2s = dcm_sampled[1,m,n]
            d3s = dcm_sampled[2,m,n]
            axis.quiver(xs, ys, zs, d1s, d2s, d3s, color = colors[m],
                        length=10000.0, normalize=True)
    
        # Dot the position of the spacecraft
        axis.scatter(xs, ys, zs, color='k')

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
arbitrary_yaw = dcmX( np.deg2rad(45.0) )
arbitrary_roll = dcmZ( np.deg2rad(45.0) )
# initial_attitude = QTR( dcm = sc.get_hill_frame().T );
initial_attitude = QTR( 
    dcm = arbitrary_yaw @ arbitrary_roll @ sc.get_hill_frame().T );

# Set the initial omegas, attitudes, and inertias.
sc.ohmBN = initial_omega
sc.attBN = initial_attitude
sc.inertia = initial_inertia

# Initialize simulation time parameters.
now, n, duration, timestep = 0.0, 0, 86400, 30.0
samples = int(duration / timestep) + 1
timeAxis = np.linspace(0, duration, samples)
sample_bigstep = 8
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
states_gtorq = np.zeros(( 3, samples ))

nBig = 0

sampleSkip = 5

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
        
    # Compute gravity gradient torque
    Rc_inertial = np.array([x[n], y[n], z[n]])
    Rc = sc.attBN.dcm.T @ Rc_inertial
    gTorque = compute_gravity_gradient_torque(sc.GM, Rc, sc.inertia)
    
    # Store the computed gravity gradient torque
    states_gtorq[:, n] = gTorque
    
    # Propagate the attitude and the angular velocity
    sc.propagate_orbit(timestep)
    sc.propagate_attitude(timestep, torque = gTorque)
    # sc.propagate_attitude(timestep, torque = [0,0,0])
    
    now += timestep
    n += 1
    
# Plot quaternions.
plt.figure()
plt.plot( timeAxis[::sampleSkip], states_quatr[0,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_quatr[1,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_quatr[2,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_quatr[3,::sampleSkip] )
plt.xlabel('Simulation time [sec]')
plt.ylabel('Body-to-Inertial Quaternions')
plt.legend(['q0','q1','q2','q3'])
plt.grid()
plt.show()

# Plot quaternions.
plt.figure()
plt.plot( timeAxis[::sampleSkip], states_gtorq[0,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_gtorq[1,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_gtorq[2,::sampleSkip] )
plt.xlabel('Simulation time [sec]')
plt.ylabel('Gravity Gradient Torque in Principal-Body Axis [N m]')
plt.legend(['$M_x$','$M_y$','$M_z$'])
plt.grid()
plt.show()
    
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
    
# Plot angular velocities.
fig2, axes2 = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
labels = [r'$\omega_{x}$', r'$\omega_{y}$', r'$\omega_{z}$']
for i, ax in enumerate(axes2):
    ax.plot( timeAxis[::sampleSkip], states_omega[i,::sampleSkip] )
    ax.set_ylabel(labels[i] + ' [rad/s]')
    ax.grid(True)
    if i == 2:
        ax.set_xlabel('Time [seconds]')
        
# Plot visualization of RTN orbit
fig3 = plt.figure(figsize=(10, 10))
axes3 = fig3.add_subplot(111, projection='3d')
plot_orbit_and_attitude(axes3,
                        x[::sampleSkip], 
                        y[::sampleSkip], 
                        z[::sampleSkip], 
                        xyz_sampled, 
                        dcm_sampled)