"""
Express a Gaussian distribution on a unit vector space. The angular distribution
is Gaussian and the unit vector is represented by a 3D vector.
"""

import numpy as np

class UnitVecGaussian:
    """
    Express a Gaussian distribution on a unit vector space. The angular 
    distribution is a 1D Gaussian and the unit vector is represented by a 3D 
    vector.

    The final result is a quaternion that represents the rotation from the
    error-free unit vector to the noisy unit vector.
    """
    def __init__(self, mean_angle = 0, std_dev_angle = 0.1):
        """
        Initialize the Gaussian distribution with the mean and standard 
        deviation of the angular distribution.

        Will only work for radians.
        """
        assert std_dev_angle >= 0
        assert std_dev_angle < np.pi
        assert mean_angle >= -np.pi
        assert mean_angle <= np.pi

        self.mean_angle = mean_angle
        self.std_dev_angle = std_dev_angle

    def _sample_unit_vec(self, n_samples = 1):
        """
        Same a unit vector uniformly on a sphere using a 3D Gaussian with
        identity covariance.
        """

        unnorm_samples = np.random.randn(n_samples, 3)
        lengths = np.linalg.norm(unnorm_samples, axis = 1)
        samples = unnorm_samples / lengths[:, np.newaxis]

        assert samples.shape == (n_samples, 3)

        return samples
    
    def _sample_angle(self, n_samples = 1):
        """
        Sample the angle from a 1D Gaussian distribution.
        """
        samples = np.random.normal(self.mean_angle, self.std_dev_angle, n_samples)

        assert samples.shape == (n_samples,)

        return samples
    
    def sample(self, n_samples, return_axis_angle = False):
        """
        We sample from the (joint) distribution of the unit vector and the 
        angle. This is what the user should call.
        """
        unit_vecs = self._sample_unit_vec(n_samples)
        angles = self._sample_angle(n_samples)

        assert unit_vecs.shape == (n_samples, 3)
        assert angles.shape == (n_samples,)

        # Make this into a quaternion
        qtr_scalar = np.cos(angles / 2)
        qtr_vec = np.sin(angles / 2)[:, np.newaxis] * unit_vecs
        qtr_arr = np.column_stack([qtr_scalar, qtr_vec])

        assert qtr_arr.shape == (n_samples, 4)

        if return_axis_angle:
            return qtr_arr, unit_vecs, angles
        else:
            return qtr_arr
