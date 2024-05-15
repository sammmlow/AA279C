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
from source.plot_orbit_and_attitude import plot_orbit_and_attitude
from source.perturbations import compute_gravity_gradient_torque

from plot_everything import plot_everything

# For saving the figures
file_path = "figures/ps5/PS5-GravityGradient-StabilityTests-"

## ===========================================================================
## TESTING OF THE ROTATIONS IN EACH CONFIGURATION AND THEIR kR, kT, kN values
## ===========================================================================

plt.close("all")

initial_inertia = np.diag( [4770.398, 6313.894, 7413.202] );

# These are the DCMs describing each of the 6 initial conditions.
deg2rad = 1.0/57.3
sc1DCM = np.identity(3)                            # Scenario 1
sc2DCM = dcmZ(90 * deg2rad)                        # Scenario 2
sc3DCM = dcmZ(90 * deg2rad) @ dcmY(-90 * deg2rad)  # Scenario 3
sc4DCM = dcmX(-90 * deg2rad)                       # Scenario 4
sc5DCM = dcmY(-90 * deg2rad) @ dcmZ(-90 * deg2rad) # Scenario 5
sc6DCM = dcmY(90 * deg2rad)                        # Scenario 6

# Print out the kR, kT, kN values as sanity checks to match Daniel's numbers
def compute_kR_kT_kN(rotation, inertia):
    I = rotation.T @ inertia @ rotation
    kR = (I[2,2] - I[1,1]) / I[0,0]
    kT = (I[2,2] - I[0,0]) / I[1,1]
    kN = (I[1,1] - I[0,0]) / I[2,2]
    print("kR = ", kR)
    print("kT = ", kT)
    print("kN = ", kN, "\n")
    return [kR, kT, kN]
    
compute_kR_kT_kN( sc1DCM, initial_inertia )
compute_kR_kT_kN( sc2DCM, initial_inertia )
compute_kR_kT_kN( sc3DCM, initial_inertia )
compute_kR_kT_kN( sc4DCM, initial_inertia )
compute_kR_kT_kN( sc5DCM, initial_inertia )
compute_kR_kT_kN( sc6DCM, initial_inertia )

## ===========================================================================
## ACTUAL CODE BEGINS HERE!
## ===========================================================================

# Initialize a spacecraft with these GEO elements.
all_rotations = [sc1DCM, sc2DCM, sc3DCM, sc4DCM, sc5DCM, sc6DCM]

alpha = 1.0 * deg2rad

# Scenario 1 should be stable regardless of yaw/pitch/roll, so include all.
pert1DCM = dcmX(alpha) @ dcmY(alpha) @ dcmZ(alpha);

# Scenario 2 should be unstable w.r.t. pitch. Perturb pitch.
pert2DCM = dcmY(alpha)

# Scenario 3 & 4 should be unstable w.r.t. yaw and roll. Perturb both.
pert3DCM = dcmX(alpha) @ dcmZ(alpha)
pert4DCM = dcmX(alpha) @ dcmZ(alpha)

# Scenario 5 & 6 should be unstable w.r.t. yaw/pitch/roll, so include all.
pert5DCM = dcmX(alpha) @ dcmY(alpha) @ dcmZ(alpha);
pert6DCM = dcmX(alpha) @ dcmY(alpha) @ dcmZ(alpha);

all_perts = [pert1DCM, pert2DCM, pert3DCM, pert4DCM, pert5DCM, pert6DCM]

for r in range(len(all_rotations)):
    
    geo_elements = [42164, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6];
    sc = Spacecraft( elements = geo_elements )
    
    each_rotation = all_rotations[r]
    each_perturbation = all_perts[r]
    
    # Setup the inertias, omegas, and initial attitudes... note the order of
    # operations of rotations! Perturb the RTN first, then snap the full
    # rotation to the perturbed RTN frame.
    dcm_rtn =  sc.get_hill_frame().T
    perturbed_dcm_rtn = each_rotation @ each_perturbation @ dcm_rtn
    initial_omega = each_rotation @ np.array([0, 0, sc.n]);
    
    # Set the initial omegas, attitudes, and inertias.
    # Perturb by +1 deg in all directions.
    sc.ohmBN = initial_omega
    sc.attBN = QTR( dcm = perturbed_dcm_rtn );
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
    states_pos   = np.zeros(( 3, samples ))
    xyz_sampled  = np.zeros(( 3, samples ))
    dcm_sampled  = np.zeros(( 3, 3, samples ))
    states_omega = np.zeros(( 3, samples ))
    states_angle = np.zeros(( 3, samples ))
    states_quatr = np.zeros(( 4, samples ))
    states_gtorq = np.zeros(( 3, samples ))
    
    # Just a counter for plotting the number of attitude triads in the 3D plot.
    nBig = 0
    
    # Make this number bigger to plot faster with fewer points.
    skip = 5
    # sampleSkip = 40
    print(f"with skip of {skip}, number of samples: ", 
          samples // skip)
    
    sample_trigger_count = 0.0;
    
    # Propagate attitude and orbit
    while now < duration:
        
        # Store the angular velocities and 321 Euler angles
        states_pos[:, n] = sc.states[0:3]
        states_omega[:, n] = sc.ohmBN
        states_angle[:, n] = sc.attBN.get_euler_angles_321()
        states_quatr[:, n] = sc.attBN.qtr
        
        # Fragile code. Will not work if time step skips this.
        if (now >= sample_trigger_count):
            xyz_sampled[:, nBig] = sc.states[0:3]
            dcm_sampled[:, :, nBig] = sc.attBN.dcm
            sample_trigger_count += sample_trigger_interval
            nBig += 1
            
        # Compute gravity gradient perturbation torques.
        gTorque = compute_gravity_gradient_torque(
            sc.GM, sc.attBN.dcm.T @ sc.states[0:3], sc.inertia)
        
        # Store the computed gravity gradient torque
        states_gtorq[:, n] = gTorque
        
        # Propagate the attitude and the angular velocity
        sc.propagate_orbit(timestep)
        sc.propagate_attitude(timestep, torque = gTorque)
        # sc.propagate_attitude(timestep, torque = [0,0,0])
        
        now += timestep
        n += 1

empty = np.array([])
plot_everything( timeAxis, skip, one_orbital_period, n_periods, file_path,
                 states_quatr, states_gtorq, empty,
                 empty, states_angle, states_omega,
                 states_pos, xyz_sampled, dcm_sampled )