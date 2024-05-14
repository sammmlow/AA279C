# -*- coding: utf-8 -*-
# Script for plotting the orbit around the Earth

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D

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