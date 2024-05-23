# We want to see how the small angle approximation impacts the
# Euler angle, quaternion, and MRP representations of attitude.
#
# At small angle, the Lie group SO(3) is approximately the Lie algebra so(3).
# But, the parameterizations of SO(3) are not all equally well-behaved.
# We want to know how far we can push the small angle approximation before
# the parameterizations break down.

import numpy as np
import matplotlib.pyplot as plt

import source.attitudes as att
import source.rotation as rot

# Parameters for saving the figures.
file_path = "figures/ps7/PS7-SmallAngles"
show_plots = False
dpi = 300
bbox_inches = 'tight'

# Test parameters.
test_vector = np.array([1, 1, 1])
test_vector = test_vector / np.linalg.norm(test_vector)


def euler_angle_rot_small_angle(phi_, theta_, psi_):
    """
    Compute the DCM for a small angle rotation.
    """
    dcm = np.array(
        [[ 1,       psi_, -theta_],
         [-psi_,    1,     phi_  ],
         [ theta_, -phi_,  1     ]]
    )
    return dcm

def qtr_small_angle(phi_, theta_, psi_):
    """
    Compute the quaternion for a small angle rotation.
    """
    qtr = np.array(
        [1, 0.5 * phi_, 0.5 * theta_, 0.5 * psi_]
    )
    return qtr

def mrp_small_angle(phi_, theta_, psi_):
    """
    Compute the MRP for a small angle rotation.
    """
    mrp = 0.25 * np.array(
        [phi_, theta_, psi_]
    )
    return mrp

# ===========================================================================
# Helper code to circumvent the normalization of quaternions in the
# source code.

def qtr_multiply(qtr1, qtr2):
    """
    Multiply two quaternions.
    """
    qtr = np.zeros(4)
    # for (r1, v1) . (r2, v2) =
    # (r1r2 - v1.v2, r1v2 + r2v1 + v1 x v2)
    r1 = qtr1[0]
    r2 = qtr2[0]
    v1 = qtr1[1:]
    v2 = qtr2[1:]

    qtr[0] = r1 * r2 - np.dot(v1, v2)
    qtr[1:] = r1 * v2 + r2 * v1 + np.cross(v1, v2)

    return qtr

def qtr_inverse(qtr):
    """
    Invert a quaternion.
    """
    qtr_conj = np.array([qtr[0], -qtr[1], -qtr[2], -qtr[3]])
    qtr_norm = np.linalg.norm(qtr)

    return qtr_conj / (qtr_norm ** 2)

def qtr_apply(qtr, vec):
    """
    Apply a quaternion to a vector.
    """
    qtr_inv = qtr_inverse(qtr)
    qtr_vec = np.array([0, *vec])
    # print("qtr    ", qtr)
    # print("qtr_inv", qtr_inv)
    # print("qtr_vec", qtr_vec)
    # print("first      :", qtr_multiply(qtr_vec, qtr))
    # print("first flip :", qtr_multiply(qtr, qtr_vec))
    # print("second     :", qtr_multiply(qtr_inv, qtr_vec))
    # print("second flip:", qtr_multiply(qtr_vec, qtr_inv))

    # vec_new = qtr_multiply(qtr, qtr_multiply(qtr_vec, qtr_inv))
    vec_new = qtr_multiply(qtr_inv, qtr_multiply(qtr_vec, qtr))
    assert np.isclose(vec_new[0], 0), f"Quaternion application failed. 0th is {vec_new[0]}"

    # print("vec_new", vec_new)

    return vec_new[1:]

# ===========================================================================

# Define the maximum angle for the small angle approximation.
max_angle = 10 * np.pi / 180

# Sample the Euler angles.
phi, theta, psi = np.random.uniform(-max_angle, max_angle, 3)

# Compute the DCMs (321).
dcmBN = rot.dcmZ(psi) @ rot.dcmY(theta) @ rot.dcmX(phi)
dcmBN_small = euler_angle_rot_small_angle(phi, theta, psi)
dcmBN_diff = dcmBN - dcmBN_small
dcmBN_diff_mag = np.linalg.norm(dcmBN_diff)
test_vector_DCM = dcmBN @ test_vector
test_vector_DCM_small = dcmBN_small @ test_vector
test_vector_DCM_diff = test_vector_DCM - test_vector_DCM_small
test_vector_DCM_diff_mag = np.linalg.norm(test_vector_DCM_diff)

print("DCM (321):")
print(dcmBN)
print("DCM (321) small angle:")
print(dcmBN_small)
print("DCM difference:")
print(dcmBN_diff)
print("DCM difference magnitude:", dcmBN_diff_mag)
print("Test vector DCM:            ", test_vector_DCM)
print("Test vector DCM small angle:", test_vector_DCM_small)
print("Test vector DCM difference: ", test_vector_DCM_diff)
print("Test vector DCM difference magnitude:", test_vector_DCM_diff_mag)

# Convert the true DCM to quaternion.
qtrBN_struct = att.QTR( dcm = dcmBN )
qtrBN = qtrBN_struct.qtr
# Cannot pass to att.QTR directly since it will normalize the quaternion.
qtrBN_small = qtr_small_angle(phi, theta, psi)
qtrBN_diff = qtrBN - qtrBN_small
qtrBN_diff_mag = np.linalg.norm(qtrBN_diff)

print("--------------------")
test_vector_QTR_stuct = qtrBN_struct.apply(test_vector)
print("MANUAL")
test_vector_QTR = qtr_apply(qtrBN, test_vector)
assert np.allclose(test_vector_QTR_stuct, test_vector_QTR), \
    f"Quaternion application failed: \nStruct: {test_vector_QTR_stuct}" + \
    f"\nManual: {test_vector_QTR}"

test_vector_QTR_small = qtr_apply(qtrBN_small, test_vector)
test_vector_QTR_diff = test_vector_QTR - test_vector_QTR_small
test_vector_QTR_diff_mag = np.linalg.norm(test_vector_QTR_diff)

test_vector_DCM_QTR_diff = test_vector_DCM - test_vector_QTR
test_vector_DCM_QTR_diff_mag = np.linalg.norm(test_vector_DCM_QTR_diff)

print()
print("Quaternion:")
print(qtrBN)
print("Quaternion small angle:")
print(qtrBN_small)
print("Quaternion difference:")
print(qtrBN_diff)
print("Quaternion difference magnitude:", qtrBN_diff_mag)
print("Test vector QTR:            ", test_vector_QTR)
print("Test vector QTR small angle:", test_vector_QTR_small)
print("Test vector QTR difference: ", test_vector_QTR_diff)
print("Test vector QTR difference magnitude:", test_vector_QTR_diff_mag)

print("Test vector DCM-QTR difference: ", test_vector_DCM_QTR_diff)
print("Test vector DCM-QTR difference magnitude:", test_vector_DCM_QTR_diff_mag)


# Convert the true DCM to MRP.
mrpBN_struct = att.MRP( dcm = dcmBN )
mrpBN = mrpBN_struct.mrp
mrpBN_small = mrp_small_angle(phi, theta, psi)
mrpBN_small_struct = att.MRP( mrp = mrpBN_small )
mrpBN_diff = mrpBN - mrpBN_small
mrpBN_diff_mag = np.linalg.norm(mrpBN_diff)

test_vector_MRP = mrpBN_struct.apply(test_vector)
test_vector_MRP_small = mrpBN_small_struct.apply(test_vector)
test_vector_MRP_diff = test_vector_MRP - test_vector_MRP_small
test_vector_MRP_diff_mag = np.linalg.norm(test_vector_MRP_diff)

test_vector_DCM_MRP_diff = test_vector_DCM - test_vector_MRP
test_vector_DCM_MRP_diff_mag = np.linalg.norm(test_vector_DCM_MRP_diff)

print("MRP:")
print(mrpBN)
print("MRP small angle:")
print(mrpBN_small)
print("MRP difference:")
print(mrpBN_diff)
print("MRP difference magnitude:", mrpBN_diff_mag)
print("Test vector MRP:            ", test_vector_MRP)
print("Test vector MRP small angle:", test_vector_MRP_small)
print("Test vector MRP difference: ", test_vector_MRP_diff)
print("Test vector MRP difference magnitude:", test_vector_MRP_diff_mag)

print("Test vector DCM-MRP difference: ", test_vector_DCM_MRP_diff)
print("Test vector DCM-MRP difference magnitude:", test_vector_DCM_MRP_diff_mag)


# Now repeat, but sweeping through a range of angles.
num_linspace = 10
phi_arr = np.linspace(-max_angle, max_angle, num_linspace)
theta_arr = np.linspace(-max_angle, max_angle, num_linspace)
psi_arr = np.linspace(-max_angle, max_angle, num_linspace)

# Direct magnitude error.
dcm_mag_error = np.zeros((num_linspace, num_linspace, num_linspace))
qtr_mag_error = np.zeros((num_linspace, num_linspace, num_linspace))
mrp_mag_error = np.zeros((num_linspace, num_linspace, num_linspace))

# Vector magnitudes.
dcm_vec_mag_error = np.zeros((num_linspace, num_linspace, num_linspace))
qtr_vec_mag_error = np.zeros((num_linspace, num_linspace, num_linspace))
mrp_vec_mag_error = np.zeros((num_linspace, num_linspace, num_linspace))

# Vector angle magnitudes.
dcm_vec_ang_error = np.zeros((num_linspace, num_linspace, num_linspace))
qtr_vec_ang_error = np.zeros((num_linspace, num_linspace, num_linspace))
mrp_vec_ang_error = np.zeros((num_linspace, num_linspace, num_linspace))

# Rotation angle magnitudes.
angle_mag = np.zeros((num_linspace, num_linspace, num_linspace))
total_angle_mag = np.zeros((num_linspace, num_linspace, num_linspace))

for i, phi in enumerate(phi_arr):
    for j, theta in enumerate(theta_arr):
        for k, psi in enumerate(psi_arr):
            angle_mag[i, j, k] = np.linalg.norm([phi, theta, psi])

            ################
            # DCM
            dcmBN = rot.dcmZ(psi) @ rot.dcmY(theta) @ rot.dcmX(phi)
            dcmBN_small = euler_angle_rot_small_angle(phi, theta, psi)
            dcmBN_diff = dcmBN - dcmBN_small
            dcm_mag_error[i, j, k] = np.linalg.norm(dcmBN_diff)

            # Test vector
            test_vector_DCM = dcmBN @ test_vector
            test_vector_DCM_small = dcmBN_small @ test_vector
            test_vector_DCM_diff = test_vector_DCM - test_vector_DCM_small
            test_vector_DCM_dot = np.dot(
                test_vector_DCM / np.linalg.norm(test_vector_DCM),
                test_vector_DCM_small / np.linalg.norm(test_vector_DCM_small)
                )

            # Vector error magnitudes and angle magnitudes.
            dcm_vec_mag_error[i, j, k] = np.linalg.norm(test_vector_DCM_diff)
            # if np.isclose(test_vector_DCM_dot, 1.0):
            #     dcm_vec_ang_error[i, j, k] = 0
            # else:
            #     dcm_vec_ang_error[i, j, k] = np.arccos(test_vector_DCM_dot)

            dcm_vec_ang_error[i, j, k] = np.arccos(test_vector_DCM_dot)

            ################
            # QTR
            qtrBN_struct = att.QTR( dcm = dcmBN )
            qtrBN = qtrBN_struct.qtr
            qtrBN_small = qtr_small_angle(phi, theta, psi)
            qtrBN_diff = qtrBN - qtrBN_small
            qtr_mag_error[i, j, k] = np.linalg.norm(qtrBN_diff)

            # Use the quaternion to get the total angle
            total_angle_mag[i, j, k] = 2 * np.arccos(qtrBN[0])

            # Test vector
            test_vector_QTR_stuct = qtrBN_struct.apply(test_vector)
            test_vector_QTR = qtr_apply(qtrBN, test_vector)
            assert np.allclose(test_vector_QTR_stuct, test_vector_QTR), \
                "Quaternion application failed: " + \
                f"\nStruct: {test_vector_QTR_stuct}" + \
                f"\nManual: {test_vector_QTR}"
            assert np.allclose(test_vector_QTR_stuct, test_vector_DCM), \
                "Quaternion application failed: " + \
                f"\nStruct: {test_vector_QTR_stuct}" + \
                f"\nDCM: {test_vector_DCM}"

            test_vector_QTR_small = qtr_apply(qtrBN_small, test_vector)
            test_vector_QTR_diff = test_vector_QTR - test_vector_QTR_small
            test_vector_QTR_dot = np.dot(
                test_vector_QTR / np.linalg.norm(test_vector_QTR),
                test_vector_QTR_small / np.linalg.norm(test_vector_QTR_small)
                )

            # Vector error magnitudes and angle magnitudes.
            qtr_vec_mag_error[i, j, k] = np.linalg.norm(test_vector_QTR_diff)
            # if np.isclose(test_vector_QTR_dot, 1.0):
            #     qtr_vec_ang_error[i, j, k] = 0
            # else:
            #     qtr_vec_ang_error[i, j, k] = np.arccos(test_vector_QTR_dot)
            qtr_vec_ang_error[i, j, k] = np.arccos(test_vector_QTR_dot)

            ################
            # MRP
            mrpBN_struct = att.MRP( dcm = dcmBN )
            mrpBN = mrpBN_struct.mrp
            mrpBN_small = mrp_small_angle(phi, theta, psi)
            mrpBN_small_struct = att.MRP( mrp = mrpBN_small )
            mrpBN_diff = mrpBN - mrpBN_small
            mrp_mag_error[i, j, k] = np.linalg.norm(mrpBN_diff)

            # Test vector
            test_vector_MRP = mrpBN_struct.apply(test_vector)
            test_vector_MRP_small = mrpBN_small_struct.apply(test_vector)

            assert np.allclose(test_vector_MRP, test_vector_DCM), \
                "MRP application failed: " + \
                f"\nMRP: {test_vector_MRP}" + \
                f"\nDCM: {test_vector_DCM}"

            test_vector_MRP_diff = test_vector_MRP - test_vector_MRP_small
            test_vector_MRP_dot = np.dot(
                test_vector_MRP / np.linalg.norm(test_vector_MRP),
                test_vector_MRP_small / np.linalg.norm(test_vector_MRP_small)
                )

            # Vector error magnitudes and angle magnitudes.
            mrp_vec_mag_error[i, j, k] = np.linalg.norm(test_vector_MRP_diff)
            # if np.isclose(test_vector_MRP_dot, 1.0):
            #     mrp_vec_ang_error[i, j, k] = 0
            # else:
            #     mrp_vec_ang_error[i, j, k] = np.arccos(test_vector_MRP_dot)
            mrp_vec_ang_error[i, j, k] = np.arccos(test_vector_MRP_dot)

# How different are the inferred angle magnitudes from the total angle?
angle_mag_flat_deg = np.rad2deg(angle_mag.flatten())
total_angle_mag_flat_deg = np.rad2deg(total_angle_mag.flatten())

# include the y=x line for reference
yxline = np.linspace(0, 2 * np.rad2deg(max_angle), 100)

fig, ax = plt.subplots()
ax.plot(angle_mag_flat_deg, total_angle_mag_flat_deg, 'o')
ax.plot(yxline, yxline, 'k--')
ax.set_xlabel('Angle magnitude (deg)')
ax.set_ylabel('Total angle magnitude (deg)')
if show_plots:
    plt.show()
fig.savefig(file_path + "-angle-magnitude-vs-total-angle-magnitude.png",
            dpi=dpi, bbox_inches=bbox_inches)

# Flatten and plot the error magnitudes compared to the angle magnitude.
def plot_vars(ax_, plot_func, angle_arr, att_arrs):
    """
    Since the plots are repetitive, we can use a function to plot them.
    """
    # Check the shapes.
    assert angle_arr.ndim == 1
    assert att_arrs.ndim == 2
    assert angle_arr.shape == att_arrs[0].shape

    plot_func(angle_arr, att_arrs[0, :], 's', label='Euler Angle DCM',
              alpha = 0.5, markeredgewidth=0.0, zorder=3)
    plot_func(angle_arr, att_arrs[1, :], 'o', label='Quaternion',
              alpha = 0.5, markeredgewidth=0.0, zorder=3)
    plot_func(angle_arr, att_arrs[2, :], '^', label='MRP',
              alpha = 0.5, markeredgewidth=0.0, zorder=3)
    ax_.legend()
    ax_.grid()

    return fig, ax

# # Convert the arrays to 2D.
# attitudes_flat = np.stack([dcm_mag_error.flatten(),
#                            qtr_mag_error.flatten(),
#                            mrp_mag_error.flatten()])

# # Plot the error magnitudes compared to the angle magnitude.
# # Make a linear plot.
# fig, ax = plt.subplots()
# fig, ax = plot_vars(ax, ax.plot, angle_mag_flat_deg, attitudes_flat)
# ax.set_xlabel('Angle magnitude (deg)')
# if show_plots:
#     plt.show()
# fig.savefig(file_path + "-angle-magnitude-linear.png",
#             dpi=dpi, bbox_inches=bbox_inches)

# # Make a semi-log plot.
# fig, ax = plt.subplots()
# fig, ax = plot_vars(ax, ax.semilogy, angle_mag_flat_deg, attitudes_flat)
# ax.set_xlabel('Angle magnitude (deg)')
# if show_plots:
#     plt.show()
# fig.savefig(file_path + "-angle-magnitude-semilog.png",
#             dpi=dpi, bbox_inches=bbox_inches)

# # Plot the error magnitudes compared to the total angle magnitude.
# # Make a linear plot.
# fig, ax = plt.subplots()
# fig, ax = plot_vars(ax, ax.plot, total_angle_mag_flat_deg, attitudes_flat)
# ax.set_xlabel('Total angle magnitude (deg)')
# if show_plots:
#     plt.show()
# fig.savefig(file_path + "-total-angle-magnitude-linear.png",
#             dpi=dpi, bbox_inches=bbox_inches)

# # Make a semi-log plot.
# fig, ax = plt.subplots()
# fig, ax = plot_vars(ax, ax.semilogy, total_angle_mag_flat_deg, attitudes_flat)
# ax.set_xlabel('Total angle magnitude (deg)')
# if show_plots:
#     plt.show()
# fig.savefig(file_path + "-total-angle-magnitude-semilog.png",
#             dpi=dpi, bbox_inches=bbox_inches)


# Convert the arrays to 2D.
attitudes_direct_flat = np.stack([
    dcm_mag_error.flatten(),
    qtr_mag_error.flatten(),
    mrp_mag_error.flatten()])
attitudes_vec_norm_flat = np.stack([
    dcm_vec_mag_error.flatten(),
    qtr_vec_mag_error.flatten(),
    mrp_vec_mag_error.flatten()])
attitudes_vec_ang_flat = np.stack([
    dcm_vec_ang_error.flatten(),
    qtr_vec_ang_error.flatten(),
    mrp_vec_ang_error.flatten()]) * 180 / np.pi

all_attitudes_flat = [attitudes_direct_flat, attitudes_vec_norm_flat,
                      attitudes_vec_ang_flat]

headers = ["direct", "vec_norm", "vec_angle"]
ylabels = ["Error magnitude ($L_2$ in Parameter Space)",
           "Error magnitude ($L_2$ in $R^3$)",
           "Error angle (deg)"]

for attitudes_flat, header, ylabel in zip(all_attitudes_flat, headers, ylabels):
    file_path_with_header = file_path + "-" + header

    # Plot the error magnitudes compared to the angle magnitude.
    # Make a linear plot.
    fig, ax = plt.subplots()
    fig, ax = plot_vars(ax, ax.plot, angle_mag_flat_deg, attitudes_flat)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Angle magnitude (deg)')
    if show_plots:
        plt.show()
    fig.savefig(file_path_with_header + "-angle-magnitude-linear.png",
                dpi=dpi, bbox_inches=bbox_inches)

    # Make a semi-log plot.
    fig, ax = plt.subplots()
    fig, ax = plot_vars(ax, ax.semilogy, angle_mag_flat_deg, attitudes_flat)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Angle magnitude (deg)')
    if show_plots:
        plt.show()
    fig.savefig(file_path_with_header + "-angle-magnitude-semilog.png",
                dpi=dpi, bbox_inches=bbox_inches)

    # Plot the error magnitudes compared to the total angle magnitude.
    # Make a linear plot.
    fig, ax = plt.subplots()
    fig, ax = plot_vars(ax, ax.plot, total_angle_mag_flat_deg, attitudes_flat)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Total angle magnitude (deg)')
    if show_plots:
        plt.show()
    fig.savefig(file_path_with_header + "-total-angle-magnitude-linear.png",
                dpi=dpi, bbox_inches=bbox_inches)

    # Make a semi-log plot.
    fig, ax = plt.subplots()
    fig, ax = plot_vars(ax, ax.semilogy, total_angle_mag_flat_deg, attitudes_flat)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Total angle magnitude (deg)')
    if show_plots:
        plt.show()
    fig.savefig(file_path_with_header + "-total-angle-magnitude-semilog.png",
                dpi=dpi, bbox_inches=bbox_inches)
