# -*- coding: utf-8 -*-

import datetime
import numpy as np
from numpy.linalg import norm

from source import iau1976
from source import ephemeris
from source import srp

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
def compute_magnetic_torque_component(t, pos_eci, att_body2eci,
                                      ncoils = 432, A = 0.0556,
                                      I = 1, debug=False):

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
    
    # Compute the magnetic moment dipole of the Earth
    deg2rad = np.pi / 180.0
    B_north_lat = 78.6 * deg2rad # Geocentric deg
    B_north_lon = 289.3 * deg2rad # Geocentric deg
    
    # Earth dipole in ECEF
    m_hat_Earth_x = np.cos(B_north_lat) * np.cos(B_north_lon)
    m_hat_Earth_y = np.cos(B_north_lat) * np.sin(B_north_lon)
    m_hat_Earth_z = np.sin(B_north_lat)
    m_hat_Earth_ecef = np.array([m_hat_Earth_x, m_hat_Earth_y, m_hat_Earth_z])
    
    # Earth dipole in ECI. Ignores pole wander (we need ERP files for that)
    N  = iau1976.nutation( t )
    S  = iau1976.diurnal( t, N )
    P  = iau1976.precession( t )
    Nt = N.transpose()
    St = S.transpose()
    Pt = P.transpose()
    m_hat_Earth_eci = Pt @ Nt @ St @ m_hat_Earth_ecef

    # Calculate the magnetic field strength at the spacecraft location
    earth_radius = 6378 # km
    pos_norm = np.linalg.norm( pos_eci )
    pos_eci_hat = pos_eci / pos_norm
    Bconstant = B0_first_order * ((earth_radius / pos_norm)**3) # T
    
    # Compute magnetic field vector in ECI
    mEarth_Rc_dot = np.dot(m_hat_Earth_eci, pos_eci_hat)
    B_modulation = (3* mEarth_Rc_dot * pos_eci_hat - m_hat_Earth_eci)  # ECI
    B_vec = (Bconstant) * B_modulation
    
    # Convert B_vec to body frame
    B_vec_body = att_body2eci.dcm.T @ B_vec

    # Calculate the magnetic moment.
    # Assume satellite dipole moment is in body-frame Z
    m_hat_body = np.array([0,0,1])
    m_max = ncoils * A * I # A m^2
    m_max_vec = m_max * m_hat_body # A m^2

    # Calculate the torque
    torque = np.cross(m_max_vec, B_vec_body)
    cross_loss = np.cross(m_hat_body, B_modulation)

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


# Computes magnetic moment torques (body). Rc in principal body.
def compute_magnetic_direction(t, pos_eci, att_body2eci):

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
    
    # Compute the magnetic moment dipole of the Earth
    deg2rad = np.pi / 180.0
    B_north_lat = 78.6 * deg2rad # Geocentric deg
    B_north_lon = 289.3 * deg2rad # Geocentric deg
    
    # Earth dipole in ECEF
    m_hat_Earth_x = np.cos(B_north_lat) * np.cos(B_north_lon)
    m_hat_Earth_y = np.cos(B_north_lat) * np.sin(B_north_lon)
    m_hat_Earth_z = np.sin(B_north_lat)
    m_hat_Earth_ecef = np.array([m_hat_Earth_x, m_hat_Earth_y, m_hat_Earth_z])
    
    # Earth dipole in ECI. Ignores pole wander (we need ERP files for that)
    N  = iau1976.nutation( t )
    S  = iau1976.diurnal( t, N )
    P  = iau1976.precession( t )
    Nt = N.transpose()
    St = S.transpose()
    Pt = P.transpose()
    m_hat_Earth_eci = Pt @ Nt @ St @ m_hat_Earth_ecef

    # Calculate the magnetic field strength at the spacecraft location
    earth_radius = 6378 # km
    pos_norm = np.linalg.norm( pos_eci )
    pos_eci_hat = pos_eci / pos_norm
    Bconstant = B0_first_order * ((earth_radius / pos_norm)**3) # T
    
    # Compute magnetic field vector in ECI
    mEarth_Rc_dot = np.dot(m_hat_Earth_eci, pos_eci_hat)
    B_modulation = (3* mEarth_Rc_dot * pos_eci_hat - m_hat_Earth_eci)  # ECI
    B_vec = (Bconstant) * B_modulation
    
    # Convert B_vec to body frame
    B_vec_body = att_body2eci.dcm.T @ B_vec

    return B_vec, B_vec_body

# All vectors in this function should be expressed in ECI frame, units in km.
def check_if_in_eclipse(pos_eci, sun_direction_eci):
    earth_radius = 6378.140
    pos_parallel = sun_direction_eci * np.dot(pos_eci, sun_direction_eci)
    pos_perpendicular = pos_eci - pos_parallel
    which_side = np.dot(pos_parallel, sun_direction_eci)
    return ((norm(pos_perpendicular) < earth_radius) and (which_side < 0))

# All vectors input in this function should be expressed in body frame.
# Returns two objects, an illumination boolean and the incident angle.
# Should run `check_if_in_eclipse` first before running this function.
def check_illumination_condition(pos_eci, att_body2eci, face_normal_body,
                                 sun_direction_eci):
    sun_direction_body = att_body2eci.dcm.T @ sun_direction_eci
    dot_product = np.dot( face_normal_body, sun_direction_body )
    boolean_illuminated = dot_product > 0.0  # Illuminated if > 0.0.
    incident_angle = np.arccos( dot_product )
    return [boolean_illuminated, incident_angle]

# This function is rather hard-coded at the moment, and assumes we only have
# four discrete objects to deal with: 02x planes (solar cells), 01x sphere,
# and 01x cylinder. Barycenters and normal vectors are all hardcoded for now.
# Each solar cell is assumed to contribute only two faces: front and back.
def compute_solar_torque_component(current_time, pos_eci, att_body2eci):
    assert type(current_time) == datetime.datetime
    
    # Initialize some total torque vector.
    total_torque = np.array([0., 0., 0.])
    
    # Get the direction vector to the sun.
    sun_pos_eci = ephemeris.compute_sun_position_eci( current_time )
    sun_direction_eci = sun_pos_eci / norm(sun_pos_eci)
    
    # If not in eclipse, return zero
    if check_if_in_eclipse(pos_eci, sun_direction_eci):
        return total_torque
    
    # Initialize the Cd/Cs/Ca. Daniel, feel free to change these values
    Cd_solar = 0.3
    Cs_solar = 0.1
    # Ca_solar = 1 - Cd_solar - Cs_solar
    Cd_titanium = 0.2
    Cs_titanium = 0.6
    # Ca_titanium = 1 - Cd_titanium - Cs_titanium
    
    # Convert from ECI to body frame
    sun_direction_body = att_body2eci.dcm.T @ sun_direction_eci
    
    # Hardcoded set of flat faces. Not the best way to do this. Also this
    # should cancel out any SRP torques since the solar cells are symmetric.
    face_areas = [
        4.00, # Solar panel left +Z
        4.00, # Solar panel left -Z
        4.00, # Solar panel right +Z
        4.00, # Solar panel right -Z
        ]
    face_normals = [
        np.array([0., 0.,  1.]), # Solar panel left +Z
        np.array([0., 0., -1.]), # Solar panel left -Z
        np.array([0., 0.,  1.]), # Solar panel right +Z
        np.array([0., 0., -1.]), # Solar panel right -Z
        ]
    face_barycenters = [
        np.array([-1.43,  3.00, 0.01]), # Solar panel left +Z
        np.array([-1.43,  3.00, 0.00]), # Solar panel left -Z
        np.array([-1.43, -3.00, 0.01]), # Solar panel right +Z
        np.array([-1.43, -3.00, 0.00]), # Solar panel right -Z
        ]
    
    # For each face, add the total radiation pressure torque
    for i in range(len(face_normals)):
        normal = face_normals[i]
        [illuminated, incident_angle] = check_illumination_condition(
            pos_eci, att_body2eci, normal, sun_direction_eci)
        if illuminated:
            area = face_areas[i]
            barycenter = face_barycenters[i]
            total_torque += np.cross(barycenter, srp.srp_area_loss_plane(
                Cs_solar, Cd_solar, sun_direction_body, normal, area))
    
    # For the curved cylinder, add the total radiation pressure torque
    cylinder_barycenter = np.array([-1.426, 0.000, -0.00314])
    cylinder_z_vec = np.array([1., 0., 0.])
    cylinder_radius = 0.625
    cylinder_height = 1.250
    total_torque += np.cross(cylinder_barycenter, srp.srp_area_loss_cylinder(
        Cs_titanium, Cd_titanium, sun_direction_body,
        cylinder_z_vec, cylinder_radius, cylinder_height))
    
    # For the fuel tank and sphere, add the total radiation pressure torque
    sphere_barycenter = np.array([0.298, 0.000, -0.00314])
    sphere_radius = 1.250
    total_torque += np.cross(sphere_barycenter, srp.srp_area_loss_sphere(
        Cd_titanium, sun_direction_body, sphere_radius))
    
    # Solar radiation pressure constant = solar constant / speed of light
    return 4.5E-6 * total_torque

