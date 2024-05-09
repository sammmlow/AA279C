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
from source.rotation import dcmX, dcmY, dcmZ

# ===========================================================================

# For saving the figures
file_path = "figures/ps5/PS5-Pert-StabilityTests-"

# ===========================================================================

# Compute gravity gradient torque. Rc in principal body frame.
def compute_gravity_gradient_torque(GM, Rc, inertia):
    RcNorm = norm(Rc)
    k = 3 * GM / (RcNorm**3) # Note that km units cancel out here.
    Rcx, Rcy, Rcz = Rc / RcNorm
    Ix, Iy, Iz = np.diag(inertia)
    Mx = Rcy * Rcz * (Iz - Iy) * k
    My = Rcx * Rcz * (Ix - Iz) * k
    Mz = Rcx * Rcy * (Iy - Ix) * k
    return np.array([Mx, My, Mz])

# Computes magnetic moment torques (body). Rc in principal body.
def compute_magnetic_torque_component(Rc, ncoils = 432, A = 0.0556, I = 1, debug=False):

    # Calculate the magnetic field overall strength (dipole)
    ref_year = 1975
    curr_year = 2024
    dyear = curr_year - ref_year
    g0_1_dot = 25.6 # nT/yr
    g1_1_dot = 10.0 # nT/yr
    h1_1_dot = -10.2 # nT/yr

    g0_1 = -30186 + g0_1_dot * dyear # nT
    g1_1 = -2036 + g1_1_dot * dyear # nT
    h1_1 = 5735 + h1_1_dot * dyear # nT

    B0_first_order = np.sqrt(g0_1**2 + g1_1**2 + h1_1**2) # nT
    B0_first_order = B0_first_order * 1E-9 # Convert to T

    # Calculate the magnetic field strength at the spacecraft location
    earth_radius = 6378 # km
    Rc_norm = np.linalg.norm( Rc )
    Rc_hat = Rc / Rc_norm
    Bconstant = B0_first_order * ((earth_radius / Rc_norm)**3) # T

    # Calculate the magnetic field with direction
    tilt_angle = np.deg2rad(11.5) # deg
    m_hat_Earth = np.array([0, np.sin(tilt_angle), np.cos(tilt_angle)])  # Assume Earth frame
    mEarth_Rc_dot = np.dot(m_hat_Earth, Rc_hat)
    B_modulation = (3* mEarth_Rc_dot * Rc_hat - m_hat_Earth)
    B_vec = (Bconstant) * B_modulation

    # Calculate the magnetic moment
    m_hat = np.array([0,0,1])  # Assume body frame
    # m_hat = np.array([0,1,0])  # Assume body frame
    m_max = ncoils * A * I # A m^2
    m_max_vec = m_max * m_hat # A m^2

    # Calculate the torque
    torque = np.cross(m_max_vec, B_vec)
    cross_loss = np.cross(m_hat, B_modulation)

    # Calculate the max case
    if debug:
        print()
        print("B at GEO [T]:      ", Bconstant)                   # MATCHES
        print("B modulation [-]:  ", np.round(B_modulation, 6))
        print("B  [T]:            ", np.round(B_vec, 10))
        print("m_max_vec [A m2]:  ", m_max_vec)                   # MATCHES
        print("torque max [N m]:  ", 2 * m_max * Bconstant)       # MATCHES
        print("torque [N m]:      ", np.round(torque, 15), np.linalg.norm(torque))
        print("cross loss [-]:    ", np.round(cross_loss, 15), np.linalg.norm(cross_loss))

    return torque

def compute_sun_position_eci(JC2000_TT):
    
    # Constants
    kDegree = np.pi / 180.0
    kArcsec = kDegree / 3600.0
    two_pi = 2.0 * np.pi
    kEarthObliquity = 23.439281 * np.pi / 180.0
    
    # Function to calculate the obliquity rotation matrix
    def obliquity_rotation_matrix():
        c = np.cos(kEarthObliquity)
        s = np.sin(kEarthObliquity)
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, c, -s],
                         [0.0, s,  c]])
    
    # Julian centuries since J2000
    T = JC2000_TT 
    
    # Mean anomaly of the Sun
    M = (357.5256 + 35999.049 * T + 1.3972 * T) % 360.0 * kDegree
    
    # Geometric mean longitude of the Sun
    l = (282.9400 * kDegree + M + 6892 * np.sin(M) * kArcsec +
         72 * np.sin(2 * M) * kArcsec) % two_pi
    
    # Distance to the Sun in meters
    r = (149.619 - 2.499 * np.cos(M) - 0.021 * np.cos(2 * M)) * 1e9
    
    # Cartesian coordinates of the Sun
    r_sun = [r * np.cos(l), r * np.sin(l), 0.0]
    
    return obliquity_rotation_matrix() * r_sun


def compute_solar_torque_component():
    return

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

initial_inertia = np.diag( [4770.398, 6313.894, 7413.202] );
    
geo_elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6];
sc = Spacecraft( elements = geo_elements )

# Setup the inertias, omegas, and initial attitudes... note the order of
# operations of rotations! Perturb the RTN first, then snap the full
# rotation to the perturbed RTN frame.
dcm_rtn =  sc.get_hill_frame().T
initial_omega = np.array([0, 0, sc.n]);

# Set the initial omegas, attitudes, and inertias.
# Perturb by +1 deg in all directions.
sc.ohmBN = initial_omega
sc.attBN = QTR( dcm = dcm_rtn );
sc.inertia = initial_inertia

# Initialize simulation time parameters.
# Orbit period is 24 hours, or 60 s/m * 60 m/hr * 24hr/day.
one_orbital_period = 60 * 60 * 24
n_periods = 10
duration = n_periods * one_orbital_period

now, n, timestep = 0.0, 0, 120.0
samples = int(duration / timestep)
print("Number of samples: ", samples)
timeAxis = np.linspace(0, duration, samples)
# sample_bigstep = 8
sample_bigstep = 11
sample_trigger_interval = duration / sample_bigstep

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
states_mtorq = np.zeros(( 3, samples ))

# Just a counter for plotting the number of attitude triads in the 3D plot.
nBig = 0

# Make this number bigger to plot faster with fewer points.
sampleSkip = 5
# sampleSkip = 40
print(f"with skip of {sampleSkip}, number of samples: ", 
      samples // sampleSkip)

sample_trigger_count = 0.0;

# Propagate attitude and orbit
while now < duration:
    
    # Store the angular velocities and 321 Euler angles
    x[n] = sc.states[0]
    y[n] = sc.states[1]
    z[n] = sc.states[2]
    states_omega[:, n] = sc.ohmBN
    states_angle[:, n] = sc.attBN.get_euler_angles_321()
    states_quatr[:, n] = sc.attBN.qtr
    
    # Fragile code. Will not work if time step skips this.
    if (now >= sample_trigger_count):
        xyz_sampled[:, nBig] = sc.states[0:3]
        dcm_sampled[:, :, nBig] = sc.attBN.dcm
        sample_trigger_count += sample_trigger_interval
        nBig += 1
        
    # Compute gravity gradient torque
    Rc_inertial = np.array([x[n], y[n], z[n]])
    Rc = sc.attBN.dcm.T @ Rc_inertial
    gTorque = compute_gravity_gradient_torque(sc.GM, Rc, sc.inertia)
    mTorque = compute_magnetic_torque_component(Rc)
    # TODO: add solar rad pressure torque
    
    # Store the computed perturbation torques
    states_gtorq[:, n] = gTorque
    states_mtorq[:, n] = mTorque
    # TODO: add solar rad pressure torque
    
    # Propagate the attitude and the angular velocity
    sc.propagate_orbit(timestep)
    sc.propagate_attitude(timestep, torque = gTorque + mTorque)
    # TODO: add solar rad pressure torque
    
    now += timestep
    n += 1
    
    
## ===========================================================================
## PLOTTING STUFF! KEEP YOUR EYES AWAY!
## ===========================================================================
    
# Plot quaternions.
plt.figure()
plt.plot( timeAxis[::sampleSkip], states_quatr[0,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_quatr[1,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_quatr[2,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_quatr[3,::sampleSkip] )
plt.xlabel('Simulation time [sec]')
plt.ylabel('Body-to-Inertial Quaternions')
plt.legend(['q0','q1','q2','q3'])

# Plot the orbital periods as vertical lines.
for i in range(n_periods + 1):
    plt.axvline(i * one_orbital_period, color='gray', linestyle='--')

plt.grid()
# plt.show()



# Save the quaternion plot
plt.savefig(file_path + 'QTR-Pert.png', dpi=200, 
            bbox_inches='tight')

# Plot gravity gradients.
plt.figure()
plt.plot( timeAxis[::sampleSkip], states_gtorq[0,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_gtorq[1,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_gtorq[2,::sampleSkip] )
plt.xlabel('Simulation time [sec]')
plt.ylabel('Gravity Gradient Torque in Principal-Body Axis [N m]')
plt.legend(['$G_x$','$G_y$','$G_z$'])

# Plot the orbital periods as vertical lines.
for i in range(n_periods + 1):
    plt.axvline(i * one_orbital_period, color='gray', linestyle='--')

plt.grid()
# plt.show()

# Save the gravity gradient plot
plt.savefig(file_path + 'gTorque-Pert.png', dpi=200, bbox_inches='tight')

# Plot magnetic moment torques.
plt.figure()
plt.plot( timeAxis[::sampleSkip], states_mtorq[0,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_mtorq[1,::sampleSkip] )
plt.plot( timeAxis[::sampleSkip], states_mtorq[2,::sampleSkip] )
plt.xlabel('Simulation time [sec]')
plt.ylabel('Magnetic Moment Torque in Principal-Body Axis [N m]')
plt.legend(['$M_x$','$M_y$','$M_z$'])

# Plot the orbital periods as vertical lines.
for i in range(n_periods + 1):
    plt.axvline(i * one_orbital_period, color='gray', linestyle='--')

plt.grid()
# plt.show()

# Save the gravity gradient plot
plt.savefig(file_path + 'mTorque-Pert.png', dpi=200, bbox_inches='tight')
    
print("Plotting Euler")

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
    
    # Plot the orbital periods as vertical lines.
    for i in range(n_periods + 1):
        ax.axvline(i * one_orbital_period, color='gray', linestyle='--')

# Save the Euler angle plot
plt.savefig(file_path + 'Angles-Pert.png', dpi=200, bbox_inches='tight')

print("Plotting angular velocities")

# Plot angular velocities.
fig2, axes2 = plt.subplots(nrows=3, ncols=1, figsize=(7, 6))
labels = [r'$\omega_{x}$', r'$\omega_{y}$', r'$\omega_{z}$']
for i, ax in enumerate(axes2):
    ax.plot( timeAxis[::sampleSkip], states_omega[i,::sampleSkip] )
    ax.set_ylabel(labels[i] + ' [rad/s]')
    ax.grid(True)
    if i == 2:
        ax.set_xlabel('Time [seconds]')
        
    # Plot the orbital periods as vertical lines.
    for i in range(n_periods + 1):
        ax.axvline(i * one_orbital_period, color='gray', linestyle='--')

# Save the angular velocity plot
plt.savefig(file_path + 'Omegas-Pert.png', dpi=200, bbox_inches='tight')
        
print("Plotting RTN")
# Plot visualization of RTN orbit
fig3 = plt.figure(figsize=(10, 10))
axes3 = fig3.add_subplot(111, projection='3d')
plot_orbit_and_attitude(axes3,
                        x[::sampleSkip], 
                        y[::sampleSkip], 
                        z[::sampleSkip], 
                        xyz_sampled, 
                        dcm_sampled)

plt.tight_layout()

# Save the RTN plot
plt.savefig(file_path + 'Orbit-Pert.png', dpi=200, bbox_inches='tight')