# -*- coding: utf-8 -*-

###############################################################################
###############################################################################
##                                                                           ##
##                                                                           ##
##    Module for computing the ephemeris of celestial objects. Useful for    ##
##    the computation of third-body gravitational forces on spacecraft and   ##
##    for computing solar radiation pressure directions in perturbation      ##
##    torque modelling.                                                      ##
##                                                                           ##
##    References:                                                            ##
##    Satellite Orbits (2010) by Montenbruck and Gill (Chapter 3)            ##
##                                                                           ##
##    Written by Samuel Y. W. Low.                                           ##
##    Last modified 10-May-2024                                              ##
##                                                                           ##
###############################################################################
###############################################################################

import datetime
import numpy as np
from math import sin, cos, pi

from source.rotation import dcmX

# Calculate the Julian centuries (including fractional) from the J2000 epoch.
# 1 Julian century = 60s * 60 * 24 * 365.25 * 100 = 3155760000s
def julian_centuries_since_J2000( target_datetime ):
    J2000_epoch = datetime.datetime(2000, 1, 1, 12, 0, 0)
    J2000_epoch_delta = target_datetime - J2000_epoch
    J2000_epoch_delta_seconds = J2000_epoch_delta.total_seconds()
    return J2000_epoch_delta_seconds / 3155760000

# Low-fidelity ephemeris of the sun in ECI. Input is a Python datetime (GPST).
def compute_sun_position_eci(dt):
    
    # Constants
    kDegree = pi / 180.0
    kArcsec = kDegree / 3600.0
    obliquity = 23.439281 * pi / 180.0;
    obliquity_matrix = dcmX(obliquity).T
    
    # Compute the Julian centuries since the J2000 epoch.
    T = julian_centuries_since_J2000( dt )
    
    # Mean anomaly of the Sun
    M = (357.5256 + 35999.049 * T + 1.3972 * T) % 360.0 * kDegree
    
    # Geometric mean longitude of the Sun
    l = (282.9400 * kDegree + M + 6892 * sin(M) * kArcsec +
         72 * sin(2 * M) * kArcsec) % 2.0 * pi
    
    # Distance to the Sun in meters
    r = (149.619 - 2.499 * cos(M) - 0.021 * cos(2 * M)) * 1e9
    
    # Cartesian coordinates of the Sun
    r_sun = np.array([r * cos(l), r * sin(l), 0.0])
    
    return obliquity_matrix @ r_sun

# Low-fidelity ephemeris of the moon in ECI. Input is a Python datetime (GPST).
def compute_moon_position_eci(dt):
    
    # Constants
    kDegree = pi / 180.0
    kArcsec = kDegree / 3600.0
    obliquity = 23.439281 * pi / 180.0;
    obliquity_matrix = dcmX(obliquity).T
    
    # Compute the Julian centuries since the J2000 epoch.
    T = julian_centuries_since_J2000( dt )

    L0 = (218.31617 + 481267.88088 * T - 1.3972 * T) % 360 * kDegree
    l = (134.96292 + 477198.86753 * T) % 360 * kDegree
    lp = (357.52543 + 35999.04944 * T) % 360 * kDegree
    F = (93.27283 + 483202.01873 * T) % 360 * kDegree
    D = (297.85027 + 445267.11135 * T) % 360 * kDegree

    lm = L0 + kArcsec * (
         22540 * sin(l) + 769 * sin(2 * l) - 4586 * sin(l - 2 * D) +
         2370 * sin(2 * D) - 668 * sin(lp) - 412 * sin(2 * F) -
         212 * sin(2 * l - 2 * D) - 206 * sin(l + lp - 2 * D) +
         192 * sin(l + 2 * D) - 165 * sin(lp - 2 * D) + 148 * sin(l - lp) -
         125 * sin(D) - 110 * sin(l + lp) - 55 * sin(2 * F - 2 * D)
    )

    bm = kArcsec * (
         18520 * sin(F + lm - L0 + (412 * sin(2 * F) + 541 * sin(lp)) 
         * kArcsec) - 526 * sin(F - 2 * D) + 44 * sin(l + F - 2 * D) -
         31 * sin(-l + F - 2 * D) - 25 * sin(-2 * l + F) -
         23 * sin(l + F - 2 * D) + 21 * sin(-l + F) + 11 * sin(-lp + F - 2 * D)
    )

    rm = (
        385000 - 20905 * cos(l) - 3699 * cos(2 * D - l) - 2956 * cos(2 * D) -
        570 * cos(2 * l) + 246 * cos(2 * l - 2 * D) - 205 * cos(lp - 2 * D) -
        171 * cos(l + 2 * D) - 152 * cos(l + lp - 2 * D)
    ) * 1e3

    r_moon = rm * np.array([cos(lm) * cos(bm), sin(lm) * cos(bm), sin(bm)])

    return obliquity_matrix * r_moon