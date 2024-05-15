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


def run_example(sensor_data, model_data):
    """
    Run all the examples
    """
    # Initialize attitude estimation module
    estimator = att_est.DeterministicAttitudeEstimator()

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
        ('ex_b', 'z', np.pi, 0.1)
    ]

    for s_ind, scenario in enumerate(scenarios):
        if scenario[0] == 'ex_a':
            _, sc_rot_axis, sc_rot_angle = scenario
            print_example_header(s_ind+1, "Orthogonal measurements")
            sense_data, ref_data = ex_a_data_orthogonal(
                sc_rot_axis, sc_rot_angle)
            run_example(sense_data, ref_data)

        elif scenario[0] == 'ex_b':
            _, sc_rot_axis, sc_rot_angle, sc_noise_std = scenario
            print_example_header(s_ind+1, "Orthogonal measurements with noise")
            sense_data, ref_data = ex_b_data_orthogonal_with_noise(
                sc_rot_axis, sc_rot_angle, sc_noise_std)
            run_example(sense_data, ref_data)

        else:
            raise ValueError(f"Invalid scenario: {scenario}")
