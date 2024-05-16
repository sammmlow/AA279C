"""
Code for the different attitude estimation algorithms
1. Deterministic algorithm (i.e., direct inverse)
2. Statistical algorithm (q-method)
"""

import numpy as np

import source.attitudes as att


def check_rotation_matrix(rot: np.ndarray, full_output: bool = False):
    """
    Check if a matrix is a rotation matrix (i.e., det = 1 and orthogonal).

    Parameters
    ----------
    rot : np.ndarray
        The 3x3 matrix to check if it is a rotation matrix.
    full_output : bool
        If True, return a tuple of the boolean values for if the matrix is a
        rotation matrix, if it is orthogonal, and if the determinant is 1.
        If False, return only if the matrix is a rotation matrix.

    Returns
    -------
    bool or tuple
    """
    # Check if the matrix is orthogonal
    is_orthogonal = np.allclose(rot @ rot.T, np.eye(3))
    # Check if the determinant is 1
    det = np.linalg.det(rot)
    is_det_one = np.isclose(det, 1.0)

    # All pass
    is_rotation_matrix = is_orthogonal and is_det_one

    if full_output:
        return is_rotation_matrix, is_orthogonal, is_det_one
    return is_rotation_matrix


def project_to_nearest_rotation(near_rot: np.ndarray) -> np.ndarray:
    """
    Project a matrix to the nearest rotation matrix using the SVD method.
    Mathematically, we are projecting onto the SO(3) manifold.

    Parameters
    ----------
    near_rot : np.ndarray
        The 3x3 matrix to project to the nearest rotation matrix.

    Returns
    -------
    np.ndarray
        The nearest rotation matrix to the input matrix.
    """
    # Perform the SVD
    umat, _, vtmat = np.linalg.svd(near_rot)
    # Calculate the rotation matrix
    rot = umat @ vtmat
    return rot



class AttitudeEstimator:
    """
    Parent class for attitude estimators, with common methods, such as verifying
    the estimate.
    """
    def __init__(self):
        pass

    def estimate(self, measurements: np.ndarray, model: np.ndarray) -> np.ndarray:
        """
        Estimate the attitude of the spacecraft given a collection of direction
        measurements.

        Parameters
        ----------
        measurements : np.array
            A 3xN array of direction measurements.
        model : np.array
            A 3xN array of model directions.

        Returns
        -------
        np.array
            The 3x3 rotation matrix from model (i.e., inertial) frame to
            measurement (i.e., body) frame.
        """
        raise NotImplementedError

    def verify_estimate(self,
                        measurements: np.ndarray,
                        model: np.ndarray,
                        estimate: np.ndarray,
                        verbose: bool = False) -> float:
        """
        Verify the estimate of the attitude of the spacecraft given a collection
        of direction measurements.

        Parameters
        ----------
        measurements : np.array
            A 3xN array of direction measurements.
        model : np.array
            A 3xN array of model directions.
        estimate : np.array
            The 3x3 rotation matrix from model (i.e., inertial) frame to
            measurement (i.e., body) frame.
        verbose : bool

        Returns
        -------
        float
            The error between the measurements and the estimated measurements.
        """
        predicted_measurements = estimate @ model

        # Compute the error as the norm of the difference between the
        # measurements and the predicted measurements, per column
        measurement_wise_error = np.linalg.norm(
            measurements - predicted_measurements, axis=0)

        total_error = np.linalg.norm(measurements - predicted_measurements)

        if verbose:
            print(f"Predicted Measurements: \n{predicted_measurements}")
            print(f"Measurements: \n{measurements}")

        return measurement_wise_error, total_error


class DeterministicAttitudeEstimator(AttitudeEstimator):
    """
    Make a deterministic (i.e., direct) estimator and call it on a stream of
    direction measurements.

    This estimator does not require an initial estimate (i.e., it is a snapshot
    estimator). It uses the direct inverse method to estimate the attitude.
    """
    def __init__(self, use_projection=True):
        self.use_projection = use_projection

    def estimate(self, measurements: np.ndarray,
                 model: np.ndarray,
                 hard_check_rot: bool = False) -> np.ndarray:
        """
        Estimate the attitude of the spacecraft given a collection of direction
        measurements.

        M = measurment matrix (3xN)
        V = model matrix (3xN)
        R = rotation matrix (3x3)

        M = R * V
        So,
        M * V^T = R * V * V^T
        R = M * V^T * (V * V^T)^-1

        Parameters
        ----------
        measurements : np.array
            A 3xN array of direction measurements. (body frame)
        model : np.array
            A 3xN array of model directions. (inertial frame)
        hard_check_rot : bool
            Call assert to check if the rotation matrix is a rotation matrix.
            Otherwise, will not check.

        Returns
        -------
        np.array
            The 3x3 rotation matrix from model (i.e., inertial) frame to
            measurement (i.e., body) frame.
        """
        # Check the shape of the measurements and model
        assert measurements.shape[0] == 3, \
            "The measurements must be a 3xN array," + \
            f" but is shape {measurements.shape}."
        assert model.shape[0] == 3, \
            f"The model must be a 3xN array, but is shape {model.shape}."
        assert measurements.shape[1] == model.shape[1], \
            "The measurements and model must have the same number" + \
            " of columns, but have shapes" + \
            f" {measurements.shape} and {model.shape}."

        # Calculate the pseudo-inverse of the model matrix
        # This is calculated as V_pinv = V^T * (V * V^T)^-1
        v_pinv = np.linalg.pinv(model)

        # Calculate the rotation matrix
        rot = measurements @ v_pinv

        if self.use_projection:
            # Project the rotation matrix to the nearest rotation matrix
            rot = project_to_nearest_rotation(rot)

        if hard_check_rot:
            # Check that the rotation matrix is orthogonal
            is_rotation_matrix = check_rotation_matrix(rot)
            assert is_rotation_matrix, \
                "The rotation matrix is not a rotation matrix." + \
                f" The matrix is \n{rot}"

        return rot


class DegenerateDeterminisitcAttitudeEstimator(AttitudeEstimator):
    """
    Special estimator for when there are only two measurements.
    """
    def __init__(self,
                 use_det_att: bool = True,
                 use_spread: bool = True,
                 verbose: bool = False):

        self.use_det_att = use_det_att

        if self.use_det_att:
            self.det_att_estimator = DeterministicAttitudeEstimator()
        else:
            # Set to None to avoid accidental use
            self.det_att_estimator = None

        self.use_spread = use_spread
        self.verbose = verbose

    def make_triad(self, directions: np.ndarray) -> np.ndarray:
        """
        Given two directions, create a triad of directions that is
        orthogonal.

        Parameters
        ----------
        directions : np.array
            A 3x2 array of directions.

        Returns
        -------
        np.array
            A 3x3 array of directions that is orthogonal.
        """
        # Check the shape of the directions; must be 3x2 arrays
        assert directions.shape == (3, 2), \
            "The directions must be a 3x2 array," + \
            f" but is shape {directions.shape}."

        if self.use_spread:
            d0_base = directions[:, 0]
            d1_base = directions[:, 1]

            # Average sum and difference
            d0 = (d0_base + d1_base) / 2
            d1 = (d0_base - d1_base) / 2

            # Normalize
            d0 /= np.linalg.norm(d0)
            d1 /= np.linalg.norm(d1)

        else:
            d0 = directions[:, 0]
            d1 = directions[:, 1]

        # The first is still the same
        p0 = d0
        # The second is the cross product of the m0 and m1
        # so that p1 is orthogonal to p0
        p1 = np.cross(d0, d1)
        p1 /= np.linalg.norm(p1) # Normalize
        # The third is the cross product of the first two
        # so that p2 is orthogonal to p0 and p1
        p2 = np.cross(p0, p1)
        p2 /= np.linalg.norm(p2) # Normalize (though it should be already)
        d_triad = np.array([p0, p1, p2]).T

        # Check the size
        assert d_triad.shape == (3, 3), \
            "The triad measurements must be a 3x3 array," + \
            f" but is shape {d_triad.shape}."

        # Check the row vs column
        assert np.allclose(d_triad[:, 0], p0), \
            "The first column of the triad directions must be the first" + \
            " directions, but is not."

        # Check that the triad is a rotation matrix
        is_rotation_matrix = check_rotation_matrix(d_triad)
        assert is_rotation_matrix, \
            "The triad is not a rotation matrix." + \
            f" The matrix is \n{d_triad}"

        return d_triad


    def estimate(self,
                 measurements: np.ndarray,
                 model: np.ndarray,
                 hard_check_rot: bool = False) -> np.ndarray:
        """
        Create three meausrements from two measurements and call the
        DeterministicAttitudeEstimator.

        Parameters
        ----------
        measurements : np.array
            A 3x2 array of direction measurements. (body frame)
        model : np.array
            A 3x2 array of model directions. (inertial frame)
        hard_check_rot : bool
            Call assert to check if the rotation matrix is a rotation matrix.
            Otherwise, will not check.

        Returns
        -------
        np.array
            The 3x3 rotation matrix from model (i.e., inertial) frame to
            measurement (i.e., body) frame.
        """
        # Check the shape of the measurements and model
        # Both must be 3x2 arrays
        assert measurements.shape == (3, 2), \
            "The measurements must be a 3x2 array," + \
            f" but is shape {measurements.shape}."
        assert model.shape == (3, 2), \
            f"The model must be a 3x2 array, but is shape {model.shape}."

        measurements_triad = self.make_triad(measurements)
        model_triad = self.make_triad(model)

        if self.use_det_att:
            if self.verbose:
                print("Using the DeterministicAttitudeEstimator")

            # Use the DeterministicAttitudeEstimator to estimate the attitude
            rot = self.det_att_estimator.estimate(
                measurements_triad, model_triad,
                hard_check_rot=hard_check_rot)

        else:
            if self.verbose:
                print("Using the Transpose method")

            # Again, we have that M = R * V
            # So, R = M * V^-1 = M * V^T
            rot = measurements_triad @ model_triad.T

        return rot


class StatisticalAttitudeEstimator(AttitudeEstimator):
    """
    Make a statistical estimator (i.e., q-method or "quaternion method"
    or "QUEST (Quaternion estimator algorithm)") and
    call it on a stream of direction measurements.
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def estimate(self, measurements: np.ndarray, model: np.ndarray,
                 weights: np.ndarray = None) -> np.ndarray:
        """
        Use the q-method to estimate the attitude of the spacecraft given a
        collection of direction measurements.

        The overall problem is to find the rotation matrix R such that
        measurements = R * model.
        (M = R * V)

        For the q-method, we are solving the following equation:
        J(q) = q.T K q - lambda q.T q
        where q is a quaternion, lambda is the Lagrange multiplier, and K is
        the following 4x4 matrix:
        [[S - I* sigma    Z    ]
         [Z^T             sigma]]
        were I = 3x3 identity, S = B + B.T, sigma = tr(B), B = WU^T, and
        Z = sum(w_i * (m_i x v_i^T)) for i in 1 to N.
        Moreover, W is a matrix with columns w_i = sqrt(weight_i) * m_i,
        U is a matrix with columns u_i = sqrt(weight_i) * v_i, and m_i and v_i
        are the ith measurements and model directions, respectively.

        The solution to the above equation is the eigenvector corresponding to
        the largest eigenvalue of K.

        Parameters
        ----------
        measurements : np.array
            A 3xN array of direction measurements.
        model : np.array
            A 3xN array of model directions.
        weights : np.array
            A 1xN array of weights for the measurements. If None, then
            all weights are 1.

        Returns
        -------
        np.array
            The 3x3 rotation matrix from model (i.e., inertial) frame to
            measurement (i.e., body) frame.
        """
        # Check the shape of the measurements and model
        num_measurements = measurements.shape[1]
        assert measurements.shape[0] == 3, \
            "The measurements must be a 3xN array," + \
            f" but is shape {measurements.shape}."
        assert model.shape[0] == 3, \
            f"The model must be a 3xN array, but is shape {model.shape}."
        assert num_measurements == model.shape[1], \
            "The measurements and model must have the same number" + \
            " of columns, but have shapes" + \
            f" {measurements.shape} and {model.shape}."

        # Check if the weights are given and are the correct shape.
        # Otherwise set to all ones.
        if weights is None:
            weights = np.ones(num_measurements)
        else:
            assert weights.shape == (num_measurements,), \
                "The weights must be a 1xN array," + \
                f" but is shape {weights.shape}."

        # Get the weights square root for the below calculations
        # We lift this to two axes to allow for broadcasting
        weights_sqrt = np.sqrt(weights)[None, :]

        # Now we need to construct K, piece by piece, starting with W and U.
        w_mat = weights_sqrt * measurements
        u_mat = weights_sqrt * model

        if self.verbose:
            print("W matrix:")
            print(w_mat)
            print("U matrix:")
            print(u_mat)

        # Check the shapes of the W and U matrices
        assert w_mat.shape == measurements.shape, \
            "The W matrix must have the same shape as the measurements." + \
            f" The shapes are {w_mat.shape} and {measurements.shape}."
        assert u_mat.shape == model.shape, \
            "The U matrix must have the same shape as the model." + \
            f" The shapes are {u_mat.shape} and {model.shape}."

        # Calculate B = WU^T
        b_mat = w_mat @ u_mat.T
        # b_mat = np.zeros((3, 3))
        # for i in range(num_measurements):
        #     b_mat += np.outer(w_mat[:, i], u_mat[:, i])

        if self.verbose:
            print("B matrix:")
            print(b_mat)

        # Check the shape of B
        assert b_mat.shape == (3, 3), \
            f"The B matrix must be a 3x3 matrix, but is shape {b_mat.shape}."

        # Calculate S = B + B.T
        s_mat = b_mat + b_mat.T

        if self.verbose:
            print("S matrix:")
            print(s_mat)

        # Calculate sigma = tr(B)
        sigma = np.trace(b_mat)

        # Above we have the block diagonals of K. Now we need the off-diagonals
        # which are Z and Z^T.
        z_mat = np.sum(
            weights_sqrt * np.cross(measurements, model, axis=0),
            axis=1)

        if self.verbose:
            print("Z matrix:")
            print(z_mat)

            for i in range(num_measurements):
                print(f"Cross product {i}:")
                print(weights_sqrt[0, i] * np.cross(measurements[:, i], model[:, i]))

        # Check the shape of Z
        assert z_mat.shape == (3,), \
            f"The Z matrix must be a 3x1 vector, but is shape {z_mat.shape}."

        # Now we can construct the K matrix
        k_mat = np.zeros((4, 4))
        k_mat[:3, :3] = s_mat - np.eye(3) * sigma
        k_mat[:3, 3] = z_mat
        k_mat[3, :3] = z_mat
        k_mat[3, 3] = sigma

        if self.verbose:
            print("K matrix:")
            print(k_mat)

        # Check that K is symmetric
        assert np.allclose(k_mat, k_mat.T), \
            f"The K matrix is not symmetric: \n{k_mat}"

        # Solve the eigenvalue problem
        eigvals, eigvecs = np.linalg.eig(k_mat)
        max_eigval_idx = np.argmax(eigvals)
        # max_eigval_idx = 0
        q_best = eigvecs[:, max_eigval_idx]
        q_best /= np.linalg.norm(q_best)

        if self.verbose:
            print("Eigenvalues:")
            print(eigvals)
            print("Eigenvectors:")
            print(eigvecs)
            print("Best Index:", max_eigval_idx)
            print("Best quaternion:")
            print(q_best)

        # Check that the quaternion is normalized
        assert np.isclose(np.linalg.norm(q_best), 1.0), \
            "The quaternion is not normalized."

        # Check the eigenvalue
        assert np.isclose(eigvals[max_eigval_idx], q_best.T @ k_mat @ q_best), \
            "The eigenvalue is not correct."

        # Convert the quaternion to a rotation matrix
        # Note that the quaternion is in the form [w, x, y, z] in the attitude
        # module, so we need to rearrange the elements. We have that
        # q = [x, y, z, w] in the code above.
        q_best_quat = att.QTR([q_best[-1], *q_best[:3]])
        rot = q_best_quat.dcm

        return rot
