# Solar Radiation Pressure Helper Functions

import numpy as np


def srp_loss_plane(cs: float, cd: float, vec_to_sun: np.ndarray,
                   normal_vector: np.ndarray):
    """
    For a flat plane, this function calculates the solar radiation pressure
    loss with respect to PA (pressure * area) due to the total specular,
    diffuse, and absorption coefficients.

    Note: c_s + c_d + c_a = 1, by definition.

    Parameters
    ----------
    cs : float
        Specular coefficient.
    cd : float
        Diffuse coefficient.
    vec_to_sun : numpy.ndarray
        Vector pointing from the spacecraft to the sun.
    normal_vector : numpy.ndarray
        Normal vector of the plane.

    Returns
    -------
    numpy.ndarray
        The dimensionless vector in the direction of SRP with magnitude
        matching the loss.
    """
    cos_theta = np.dot(vec_to_sun, normal_vector)

    s_direction_component = (1 - cs) * vec_to_sun
    n_direction_component = 2 * (cs * cos_theta + (1/3) * cd) * normal_vector

    # Check that the components are 3D vectors.
    assert s_direction_component.shape == (3,), s_direction_component
    assert n_direction_component.shape == (3,), n_direction_component

    # Negative sign ensures that the SRP is pushing the spacecraft away from
    # the sun.
    total_srp_loss = -cos_theta * (s_direction_component + n_direction_component)

    assert total_srp_loss.shape == (3,), total_srp_loss

    return total_srp_loss


def srp_area_loss_plane(cs: float, cd: float, vec_to_sun: np.ndarray,
                        normal_vector: np.ndarray,
                        area_plane: float):
    """
    Wraps the srp_loss_plane function to include the area of the plane.

    Parameters
    ----------
    cs : float
        Specular coefficient.
    cd : float
        Diffuse coefficient.
    vec_to_sun : numpy.ndarray
        Vector pointing from the spacecraft to the sun.
    normal_vector : numpy.ndarray
        Normal vector of the plane.
    area_plane : float
        Area of the plane.

    Returns
    -------
    numpy.ndarray
        The effective area vector in the direction of SRP with magnitude
        matching the loss including area contribution.
    """
    return srp_loss_plane(cs, cd, vec_to_sun, normal_vector) * area_plane


def srp_loss_sphere(cd: float, vec_to_sun: np.ndarray):
    """
    For a sphere, this function calculates the solar radiation pressure loss
    with respect to PA (pressure * area) due to the total specular, diffuse,
    and absorption coefficients.

    Note: c_s + c_d + c_a = 1, by definition.

    Parameters
    ----------
    cd : float
        Diffuse coefficient.
    vec_to_sun : numpy.ndarray
        Vector pointing from the spacecraft to the sun.

    Returns
    -------
    numpy.ndarray
        The dimensionless vector in the direction of SRP with magnitude
        matching the loss.
    """
    # Check that the components are 3D vectors.
    assert vec_to_sun.shape == (3,), vec_to_sun

    # Negative sign ensures that the SRP is pushing the spacecraft away from
    # the sun.
    total_srp_loss = -((1/4) + (1/9) * cd) * vec_to_sun

    assert total_srp_loss.shape == (3,), total_srp_loss

    return total_srp_loss


def srp_area_loss_sphere(cd: float, vec_to_sun: np.ndarray,
                         radius: float):
    """
    Wraps the srp_loss_sphere function to include the area of the sphere.

    Parameters
    ----------
    cd : float
        Diffuse coefficient.
    vec_to_sun : numpy.ndarray
        Vector pointing from the spacecraft to the sun.
    radius : float
        Radius of the sphere.

    Returns
    -------
    numpy.ndarray
        The effective area vector in the direction of SRP with magnitude
        matching the loss including area contribution.
    """
    surface_area = 4 * np.pi * radius**2
    return srp_loss_sphere(cd, vec_to_sun) * surface_area



def srp_area_loss_cylinder(cs: float, cd: float, vec_to_sun: np.ndarray,
                        z_vector: np.ndarray, radius: float, height: float):
    """
    For a cylinder, this function calculates the solar radiation pressure loss
    with respect to PA (pressure * area) due to the total specular, diffuse,
    and absorption coefficients.

    Note: c_s + c_d + c_a = 1, by definition.

    Unlike for the plane and sphere, the cylinder area is not multiplicative
    throughout, so we need the radius and height to directly.

    Parameters
    ----------
    cs : float
        Specular coefficient.
    cd : float
        Diffuse coefficient.
    vec_to_sun : numpy.ndarray
        Vector pointing from the spacecraft to the sun.
    z_vector : numpy.ndarray
        The axis of the cylinder.

    Returns
    -------
    numpy.ndarray
        The effective area vector in the direction of SRP with magnitude
        matching the loss including area contribution.
    """

    # To avoid confusion we use psi for the angle between the sun vector and
    # the cylinder axis, rather than theta (which is for the plane normal).
    cos_psi = np.dot(vec_to_sun, z_vector)
    sin_psi = np.sqrt(1 - cos_psi**2)

    # We have two components A_s is in the specular direction and
    # A_z is in the cylinder axis direction.
    # Each has two separate components for the face versus lateral parts.

    as_lateral = (sin_psi * (1 + (1/3) * cs)) + (np.pi / 6) * cd
    as_lateral *= 2 * radius * height
    as_face = (1 - cs) * cos_psi
    as_face *= np.pi * radius**2

    az_lateral = ((-4/3) * cs * sin_psi) - (np.pi / 6) * cd
    az_lateral *= 2 * radius * height
    az_face = 2 * (cs * cos_psi + (1/3) * cd)
    az_face *= np.pi * radius**2

    total_sun_direction = (as_lateral + as_face) * vec_to_sun
    total_z_direction = (az_lateral + az_face) * cos_psi * z_vector

    # Check that the components are 3D vectors.
    assert total_sun_direction.shape == (3,), total_sun_direction
    assert total_z_direction.shape == (3,), total_z_direction

    # Negative sign ensures that the SRP is pushing the spacecraft away from
    # the sun.
    total_srp_area_loss = -cos_psi * (total_sun_direction + total_z_direction)

    assert total_srp_area_loss.shape == (3,), total_srp_area_loss

    return total_srp_area_loss
