# Moment of Inertia (MOI) utilities

import numpy as np

def moi_cuboid(mass, width, length, height):
    """
    Calculate the moment of inertia for a cuboid.

    Parameters
    ----------
    mass : float
        Mass of the cuboid in kg.
    width : float
        Width of the cuboid in m. Along x-axis.
    length : float
        Length of the cuboid in m. Along y-axis.
    height : float
        Height of the cuboid in m. Along z-axis.

    Returns
    -------
    float
        Moment of inertia of the cuboid in kg m^2.
    """
    coeff = mass / 12

    Ix = (length**2 + height**2)
    Iy = (width**2 + height**2)
    Iz = (width**2 + length**2)

    moi = coeff * np.diag([Ix, Iy, Iz])

    return moi

def moi_cylinder(mass, radius, height, axis=0):
    """
    Calculate the moment of inertia for a cylinder.

    Parameters
    ----------
    mass : float
        Mass of the cylinder in kg.
    radius : float
        Radius of the cylinder in m.
    height : float
        Height of the cylinder in m.
    axis : int, optional
        Axis of cylinder. Default is 1 (i.e., x).

    Returns
    -------
    float
        Moment of inertia of the cylinder in kg m^2.
    """

    Irect = (1/12) * mass * (3*radius**2 + height**2)
    Icirc = (1/2) * mass * radius**2

    I_vec = [Irect] * 3
    I_vec[axis] = Icirc

    moi = np.diag(I_vec)

    return moi

def moi_solid_sphere(mass, radius):
    """
    Calculate the moment of inertia for a solid sphere.

    Parameters
    ----------
    mass : float
        Mass of the sphere in kg.
    radius : float
        Radius of the sphere in m.

    Returns
    -------
    float
        Moment of inertia of the sphere in kg m^2.
    """

    return (2/5) * mass * (radius**2) * np.eye(3)


def moi_hollow_sphere(mass, radius):
    """
    Calculate the moment of inertia for a hollow sphere.

    Parameters
    ----------
    mass : float
        Mass of the hollow sphere in kg.
    radius : float
        Outer radius of the hollow sphere in m.

    Returns
    -------
    float
        Moment of inertia of the hollow sphere in kg m^2.
    """

    return (2/3) * mass * (radius**2) * np.eye(3)

#################################
#################################
#################################

def moi_tensor_parallel_axis(moi, mass, pos_vec):
    """
    Calculate the moment of inertia (moi) tensor about a new point where
    the vector from the new point to the old point is pos_vec.

    I_new = I_old + m * (pos_vec.pos_vec * I - pos_vec * pos_vec^T)

    Parameters
    ----------
    moi : ndarray
        The original moment of inertia tensor (3x3).
    mass : float
        Mass of the object.
    pos_vec : ndarray
        The vector from the new point to the old point (3x1).

    Returns
    -------
    ndarray
        The new moment of inertia tensor (3x3).
    """
    # Check the shapes are correct
    assert moi.shape == (3, 3)
    assert pos_vec.shape == (3,)

    # Check that the moi is symmetric
    assert np.allclose(moi, moi.T)

    # Calculate the new moment of inertia tensor
    diag_term = np.dot(pos_vec, pos_vec) * np.eye(3)
    outer_term = np.outer(pos_vec, pos_vec)

    # Put it all together
    moi_new = moi + mass * (diag_term - outer_term)

    # Check the new moi is symmetric
    assert np.allclose(moi_new, moi_new.T)

    return moi_new
