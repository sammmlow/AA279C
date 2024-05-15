# Test out the attitude estimation code on a simple example.

import numpy as np

import source.attitude_estimation as att_est
import source.rotation as rotation
# import source.attitudes as att


def ex_a_data_orthogonal(rot_axis, rot_angle):
    """
    Simple 3 orthogonal direction measurements. Should straight forward.
    """
    # Check that the rotation angle is in radians
    if rot_angle < 0 or rot_angle > 2*np.pi:
        raise ValueError("Rotation angle must be in radians. " + \
                         f"Given: {rot_angle}")

    if rot_axis == 'x':
        rot_data = rotation.dcmX(rot_angle)
    elif rot_axis == 'y':
        rot_data = rotation.dcmY(rot_angle)
    elif rot_axis == 'z':
        rot_data = rotation.dcmZ(rot_angle)
    else:
        raise ValueError(f'Invalid rotation axis. Given: {rot_axis}')

    # Reference data
    model_data = np.eye(3)

    return rot_data, model_data


def ex_b_data_orthogonal_with_noise(rot_axis, rot_angle, noise_std=0.1):
    """
    Uses the example A data, but adds noise to the measurements.
    """
    # Get the noiseless data
    rot_data, model_data = ex_a_data_orthogonal(rot_axis, rot_angle)

    # Add noise to the measurements
    rot_data += np.random.normal(0, noise_std, rot_data.shape)

    # Renormalize the measurements (each column is a unit vector)
    rot_data /= np.linalg.norm(rot_data, axis=0)

    return rot_data, model_data


def ex_c_data_two_measure(ref_axis='x', rng_seed=0, cos_angle_limit=0.1):
    """
    Two measurements that are not orthogonal.
    """
    # Set the random seed
    np.random.seed(rng_seed)

    # Generate two random vectors
    vec1 = np.random.rand(3)
    vec1 /= np.linalg.norm(vec1)

    # Generate until the second vector is at least cos_angle_limit
    # away from the first vector and at least cos_angle_limit away
    # from being orthogonal to the first vector.
    while True:
        vec2 = np.random.rand(3)
        # Normalize the vectors
        vec2 /= np.linalg.norm(vec2)

        # Check the angle between the vectors
        cos_angle = np.dot(vec1, vec2)
        if (np.abs(cos_angle) > cos_angle_limit) and \
            (np.abs(cos_angle) < 1-cos_angle_limit):
            break

    # Sensor data
    sensor_data = np.array([vec1, vec2]).T

    # Check the size
    assert sensor_data.shape == (3, 2), \
        "The sensor data must be a 3x2 array," + \
        f" but is shape {sensor_data.shape}."

    # # Reference data
    # # First vector is along the desired axis
    # if ref_axis == 'x':
    #     ref_vec1 = np.array([1, 0, 0])
    #     dcm_func = rotation.dcmZ
    # elif ref_axis == 'y':
    #     ref_vec1 = np.array([0, 1, 0])
    #     dcm_func = rotation.dcmX
    # elif ref_axis == 'z':
    #     ref_vec1 = np.array([0, 0, 1])
    #     dcm_func = rotation.dcmY
    # else:
    #     raise ValueError(f'Invalid rotation axis. Given: {ref_axis}')

    # # Second vector is an angle away from the first vector that matches
    # # the angle between the two random vectors (vec1 and vec2).
    # # This is to make the reference data match the model data.
    # angle_to_rotate = np.arccos(cos_angle)
    # print(f"Angle between the vectors: {angle_to_rotate} (rad)")
    # print(f"Angle between the vectors: {angle_to_rotate*180/np.pi} (deg)")
    # ref_vec2 = dcm_func(angle_to_rotate) @ ref_vec1

    # model_data = np.array([ref_vec1, ref_vec2]).T
    # # Check the size
    # assert model_data.shape == (3, 2), \
    #     "The sensor data must be a 3x2 array," + \
    #     f" but is shape {model_data.shape}."

    # # Check that the dot product is the same
    # print("Dot products:")
    # print(np.dot(sensor_data[:, 0], sensor_data[:, 1]))
    # print(np.dot(model_data[:, 0], model_data[:, 1]))
    # assert np.abs(np.dot(sensor_data[:, 0], sensor_data[:, 1]) - \
    #               np.dot(model_data[:, 0], model_data[:, 1])) < 1e-10, \
    #     "The dot product of the sensor data and model data are not the same."

    # print("Sensor data:")
    # print(sensor_data)
    # print("Model data:")
    # print(model_data)
    # print()

    # Try just rotating the sensor data to match the model data
    sensor_to_model = rotation.dcmZ(np.pi/2)
    print()
    print("Sensor to model DCM:")
    print(sensor_to_model)

    model_data = sensor_to_model @ sensor_data

    return sensor_data, model_data



def run_example(estimator, sensor_data, model_data):
    """
    Run all the examples
    """
    # Estimate attitude
    attitude = estimator.estimate(sensor_data, model_data)

    # Print the estimated attitude
    print("Estimated attitude:")
    print(attitude)

    # Verify that the estimated attitude is a rotation matrix
    is_rotation_matrix, is_orthogonal, is_det_one = \
        att_est.check_rotation_matrix(attitude, full_output=True)
    print(f"Is rotation matrix: {is_rotation_matrix}")
    print(f"Is orthogonal: {is_orthogonal}")
    print(f"Is determinant 1: {is_det_one}")

    # Check the error
    att_errors = estimator.verify_estimate(sensor_data, model_data, attitude)
    print(f"Total error: {att_errors[1]}")
    print(f"Error per measurement: {att_errors[0]}")


def print_example_header(example_num, example_name):
    """
    Print the header for the example
    """
    print("\n----------")
    print(f"Example {example_num}: {example_name}")
    print("----------")


if __name__ == "__main__":

    scenarios = [
        ('ex_a', 'x', np.pi/2),
        ('ex_a', 'y', np.pi/4),
        ('ex_a', 'z', np.pi),
        ('ex_b', 'x', np.pi/2, 0.001),
        ('ex_b', 'y', np.pi/4, 0.001),
        ('ex_b', 'z', np.pi, 0.001),
        ('ex_b', 'x', np.pi/2, 0.1),
        ('ex_b', 'y', np.pi/4, 0.1),
        ('ex_b', 'z', np.pi, 0.1),
        ('ex_c', 'x', 0, 0.1, True),
        ('ex_c', 'y', 0, 0.1, True),
        ('ex_c', 'z', 0, 0.1, True),
        ('ex_c', 'x', 0, 0.1, False),
        ('ex_c', 'y', 0, 0.1, False),
        ('ex_c', 'z', 0, 0.1, False)
    ]

    for s_ind, scenario in enumerate(scenarios):
        if scenario[0] == 'ex_a':
            _, sc_rot_axis, sc_rot_angle = scenario
            print_example_header(s_ind+1,
                                 f"Orthogonal measurements in {sc_rot_axis}")

            # Instantiate the estimator
            estimator_under_test = att_est.DeterministicAttitudeEstimator()

            # Get the data
            sense_data, ref_data = ex_a_data_orthogonal(
                sc_rot_axis, sc_rot_angle)

            # Run the example
            run_example(estimator_under_test, sense_data, ref_data)

        elif scenario[0] == 'ex_b':
            _, sc_rot_axis, sc_rot_angle, sc_noise_std = scenario
            print_example_header(s_ind+1,
                f"Orthogonal measurements in {sc_rot_axis} with noise {sc_noise_std}")

            # Instantiate the estimator
            estimator_under_test = att_est.DeterministicAttitudeEstimator()

            # Get the data
            sense_data, ref_data = ex_b_data_orthogonal_with_noise(
                sc_rot_axis, sc_rot_angle, sc_noise_std)

            # Run the example
            run_example(estimator_under_test, sense_data, ref_data)

        elif scenario[0] == 'ex_c':
            _, sc_ref_axis, sc_rng, sc_cos_angle_limit, sc_use_det = scenario
            print_example_header(s_ind+1,
                f"Two measurements in {sc_ref_axis}, using Det Att? {sc_use_det}")

            # Instantiate the estimator
            estimator_under_test = \
                att_est.DegenerateDeterminisitcAttitudeEstimator(
                    use_det_att=sc_use_det)

            # Get the data
            sense_data, ref_data = ex_c_data_two_measure(
                sc_ref_axis, rng_seed=sc_rng,
                cos_angle_limit=sc_cos_angle_limit)

            # Run the example
            run_example(estimator_under_test, sense_data, ref_data)

        else:
            raise ValueError(f"Invalid scenario: {scenario}")
