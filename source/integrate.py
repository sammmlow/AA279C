# -*- coding: utf-8 -*-

###############################################################################
###############################################################################
##                                                                           ##
##      ___  _   _   __   ____  ____   __   _   _ _____                      ##
##     / _ \| | | | /  \ |  _ \| __ \ /  \ | \ | |_   _|                     ##
##    ( |_| ) |_| |/ /\ \| |_| | -/ // /\ \|  \| | | |                       ##
##     \_  /|_____| /--\ |____/|_|\_\ /--\ |_\___| |_|                       ##
##       \/                                               v 0.0              ##
##                                                                           ##
##    Vanilla-flavoured Runge-Kutta (Simpson's) 4th Order Integrator         ##
##                                                                           ##
##    Written by Samuel Y. W. Low.                                           ##
##    First created 17-Dec-2021 14:36 PM (+8 GMT)                            ##
##    Last modified 10-Apr-2023 20:33 PM (-8 GMT)                            ##
##                                                                           ##
###############################################################################
###############################################################################

import numpy as np
from copy import copy

from source import forces

def integrate_orbit( sc, dt ):
    '''
    Orbit propagator for one step, using RK4 (3/8 Rule).
    Updates a spacecraft in-place.
    
    Parameters
    ----------
    sc : spacecraft
        Spacecraft object (typically a spacecraft calls itself)
    dt : integer
        Time step size (s)

    Returns
    -------
    sc : spacecraft
        Spacecraft object with updated states.
    '''
    
    c = 1.0/3.0
    pos, vel = np.array(sc.states[:3]), np.array(sc.states[3:])
    
    # K1
    k1p = vel
    k1v = forces.forces( pos, vel, sc )
    
    # K2
    k2p = vel + dt * (c*k1v)
    k2v = forces.forces( pos + dt*(c*k1p), vel + dt*(c*k1v), sc )
    
    # K3
    k3p = vel + dt * (k2v-c*k1v)
    k3v = forces.forces( pos + dt*(k2p-c*k1p), vel + dt*(k2v-c*k1v), sc )
    
    # K4
    k4p = vel + dt * (k1v-k2v+k3v)
    k4v = forces.forces( pos + dt*(k1p-k2p+k3p), vel + dt*(k1v-k2v+k3v), sc )
    
    # Simpson's Rule variant to RK4 update step
    posf = pos + (dt/8) * (k1p + 3*k2p + 3*k3p + k4p)
    velf = vel + (dt/8) * (k1v + 3*k2v + 3*k3v + k4v)
    
    sc.states = list(posf) + list(velf)
    
    return sc

def integrate_attitude( sc, dt, torque ):
    '''
    Attitude propagator for one step, using the standard RK4.
    Accepts an external torque. Updates a spacecraft in-place.
    
    Parameters
    ----------
    sc : spacecraft
        Spacecraft object (typically a spacecraft calls itself)
    dt : integer
        Time step size (s)

    Returns
    -------
    sc : spacecraft
        Spacecraft object with updated ohmBN and attBN states.
    '''
    
    def ohmDotBN(inertia, ohmBN, torque):
        inertia_inverse = np.linalg.inv( inertia )
        gyroscopic = np.cross( ohmBN, inertia @ ohmBN )
        return inertia_inverse @ ( torque - gyroscopic )
    
    def qtrDotBN(qtr, ohm):
        qM = np.array([[-1*qtr[1], -1*qtr[2], -1*qtr[3]],
                       [   qtr[0], -1*qtr[3],    qtr[2]],
                       [   qtr[3],    qtr[0], -1*qtr[1]],
                       [-1*qtr[2],    qtr[1],    qtr[0]]])
        return 0.5 * np.transpose( qM @ ohm )
    
    def crpDotBN(crp, ohm):
        crp_matrix = np.zeros((3,3))
        crp_matrix[0,0] = 1 + crp[0]**2
        crp_matrix[0,1] = crp[0] * crp[1] - crp[2]
        crp_matrix[0,2] = crp[0] * crp[2] + crp[1]
        crp_matrix[1,0] = crp[1] * crp[0] + crp[2]
        crp_matrix[1,1] = 1 + crp[1]**2
        crp_matrix[1,2] = crp[1] * crp[2] - crp[0]
        crp_matrix[2,0] = crp[2] * crp[0] - crp[1]
        crp_matrix[2,1] = crp[2] * crp[1] + crp[0]
        crp_matrix[2,2] = 1 + crp[2]**2
        return 0.5 * np.transpose( crp_matrix @ ohm )
    
    def mrpDotBN(mrp, ohm):
        mt = np.array([[      0.0, -1*mrp[2],    mrp[1]],
                       [   mrp[2],       0.0, -1*mrp[0]],
                       [-1*mrp[1],    mrp[0],      0.0]])
        B = (( 1 - np.dot(mrp, mrp) ) * np.eye(3))
        B = B + (2*mt) + 2 * np.outer(mrp,mrp)
        return 0.25 * np.transpose( B @ ohm )
    
    # Define a half time step.
    cdt = 0.5 * dt
    
    # Note: ODE for angular velocities does not depend on time or attitude.
    ohmBN_init = copy(sc.ohmBN)
    k1w = ohmDotBN( sc.inertia, ohmBN_init, torque )
    k2w = ohmDotBN( sc.inertia, ohmBN_init + cdt * k1w, torque )
    k3w = ohmDotBN( sc.inertia, ohmBN_init + cdt * k2w, torque )
    k4w = ohmDotBN( sc.inertia, ohmBN_init + dt * k3w, torque )
    
    sc.ohmBN = ohmBN_init + dt * (k1w + 2*k2w + 2*k3w + k4w) / 6.0
    
    # Check if the coordinate type is a quaternion.
    if sc.attBN.strID() == 'QTR' and sc.attBR.strID() == 'QTR':
        if sc.attBN[0] < 0.0:
            sc.attBN.qtr = -1 * sc.attBN.qtr # Fix long/short rotation
        
        qBN_init = copy(sc.attBN)
        k1q = qtrDotBN( qBN_init, ohmBN_init )
        k2q = qtrDotBN( qBN_init + cdt * k1q, ohmBN_init + cdt * k1w)
        k3q = qtrDotBN( qBN_init + cdt * k2q, ohmBN_init + cdt * k2w )
        k4q = qtrDotBN( qBN_init + dt * k3q, ohmBN_init + dt * k3w )
        
        sc.attBN.qtr = qBN_init.qtr + dt * (k1q + 2*k2q + 2*k3q + k4q) / 6.0
        sc.attBN.normalise()
        
    # Check if the coordinate type is a classical Rodrigues parameter.
    if sc.attBN.strID() == 'CRP' and sc.attBR.strID() == 'CRP':
        
        cBN_init = copy(sc.attBN)
        k1c = crpDotBN( cBN_init, ohmBN_init )
        k2c = crpDotBN( cBN_init + cdt * k1c, ohmBN_init + cdt * k1w)
        k3c = crpDotBN( cBN_init + cdt * k2c, ohmBN_init + cdt * k2w )
        k4c = crpDotBN( cBN_init + dt * k3c, ohmBN_init + dt * k3w )
        
        sc.attBN.crp = cBN_init.crp + dt * (k1c + 2*k2c + 2*k3c + k4c) / 6.0
        
    # Check if the coordinate type is a modified Rodrigues parameter.
    if sc.attBN.strID() == 'MRP' and sc.attBR.strID() == 'MRP':
        
        mBN_init = copy(sc.attBN)
        k1m = mrpDotBN( mBN_init, ohmBN_init )
        k2m = mrpDotBN( mBN_init + cdt * k1m, ohmBN_init + cdt * k1w)
        k3m = mrpDotBN( mBN_init + cdt * k2m, ohmBN_init + cdt * k2w )
        k4m = mrpDotBN( mBN_init + dt * k3m, ohmBN_init + dt * k3w )
        
        sc.attBN.mrp = mBN_init.mrp + dt * (k1m + 2*k2m + 2*k3m + k4m) / 6.0
        
        if np.linalg.norm(sc.attBN.mrp) > 1.0:
            sc.attBN.set_mrp_shadow()
    
    return sc
