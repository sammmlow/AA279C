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

def quaternion_small_angle(phi_, theta_, psi_):
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

# Define the maximum angle for the small angle approximation.
max_angle = 10 * np.pi / 180

# Sample the Euler angles.
phi, theta, psi = np.random.uniform(-max_angle, max_angle, 3)

# Compute the DCMs (321).
dcmBN = rot.dcmZ(psi) @ rot.dcmY(theta) @ rot.dcmX(phi)
dcmBN_small = euler_angle_rot_small_angle(phi, theta, psi)
dcmBN_diff = dcmBN - dcmBN_small
dcmBN_diff_mag = np.linalg.norm(dcmBN_diff)

print("DCM (321):")
print(dcmBN)
print("DCM (321) small angle:")
print(dcmBN_small)
print("DCM difference:")
print(dcmBN_diff)
print("DCM difference magnitude:", dcmBN_diff_mag)

# Convert the true DCM to quaternion.
qtrBN = att.QTR( dcm = dcmBN ).qtr
# Cannot pass to att.QTR directly since it will normalize the quaternion.
qtrBN_small = quaternion_small_angle(phi, theta, psi)
qtrBN_diff = qtrBN - qtrBN_small
qtrBN_diff_mag = np.linalg.norm(qtrBN_diff)

print("Quaternion:")
print(qtrBN)
print("Quaternion small angle:")
print(qtrBN_small)
print("Quaternion difference:")
print(qtrBN_diff)
print("Quaternion difference magnitude:", qtrBN_diff_mag)


# Convert the true DCM to MRP.
mrpBN = att.MRP( dcm = dcmBN ).mrp
mrpBN_small = mrp_small_angle(phi, theta, psi)
mrpBN_diff = mrpBN - mrpBN_small
mrpBN_diff_mag = np.linalg.norm(mrpBN_diff)

print("MRP:")
print(mrpBN)
print("MRP small angle:")
print(mrpBN_small)
print("MRP difference:")
print(mrpBN_diff)
print("MRP difference magnitude:", mrpBN_diff_mag)


# Now repeat, but sweeping through a range of angles.
num_linspace = 25
phi_arr = np.linspace(-max_angle, max_angle, num_linspace)
theta_arr = np.linspace(-max_angle, max_angle, num_linspace)
psi_arr = np.linspace(-max_angle, max_angle, num_linspace)

dcm_mag_error = np.zeros((num_linspace, num_linspace, num_linspace))
qtr_mag_error = np.zeros((num_linspace, num_linspace, num_linspace))
mrp_mag_error = np.zeros((num_linspace, num_linspace, num_linspace))

angle_mag = np.zeros((num_linspace, num_linspace, num_linspace))
total_angle_mag = np.zeros((num_linspace, num_linspace, num_linspace))

for i, phi in enumerate(phi_arr):
    for j, theta in enumerate(theta_arr):
        for k, psi in enumerate(psi_arr):
            angle_mag[i, j, k] = np.linalg.norm([phi, theta, psi])

            dcmBN = rot.dcmZ(psi) @ rot.dcmY(theta) @ rot.dcmX(phi)
            dcmBN_small = euler_angle_rot_small_angle(phi, theta, psi)
            dcmBN_diff = dcmBN - dcmBN_small
            dcm_mag_error[i, j, k] = np.linalg.norm(dcmBN_diff)

            qtrBN = att.QTR( dcm = dcmBN ).qtr
            qtrBN_small = quaternion_small_angle(phi, theta, psi)
            qtrBN_diff = qtrBN - qtrBN_small
            qtr_mag_error[i, j, k] = np.linalg.norm(qtrBN_diff)

            # Use the quaternion to get the total angle
            total_angle_mag[i, j, k] = 2 * np.arccos(qtrBN[0])

            mrpBN = att.MRP( dcm = dcmBN ).mrp
            mrpBN_small = mrp_small_angle(phi, theta, psi)
            mrpBN_diff = mrpBN - mrpBN_small
            mrp_mag_error[i, j, k] = np.linalg.norm(mrpBN_diff)

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

    plot_func(angle_arr, att_arrs[0, :], 'o', label='Euler Angle DCM')
    plot_func(angle_arr, att_arrs[1, :], 'x', label='Quaternion')
    plot_func(angle_arr, att_arrs[2, :], '^', label='MRP')
    ax_.set_ylabel('Error magnitude ($L_2$ norm of difference)')
    ax_.legend()
    ax_.grid()

    return fig, ax

# Convert the arrays to 2D.
attitudes_flat = np.stack([dcm_mag_error.flatten(),
                           qtr_mag_error.flatten(),
                           mrp_mag_error.flatten()])

# Plot the error magnitudes compared to the angle magnitude.
# Make a linear plot.
fig, ax = plt.subplots()
fig, ax = plot_vars(ax, ax.plot, angle_mag_flat_deg, attitudes_flat)
ax.set_xlabel('Angle magnitude (deg)')
if show_plots:
    plt.show()
fig.savefig(file_path + "-angle-magnitude-linear.png",
            dpi=dpi, bbox_inches=bbox_inches)

# Make a semi-log plot.
fig, ax = plt.subplots()
fig, ax = plot_vars(ax, ax.semilogy, angle_mag_flat_deg, attitudes_flat)
ax.set_xlabel('Angle magnitude (deg)')
if show_plots:
    plt.show()
fig.savefig(file_path + "-angle-magnitude-semilog.png",
            dpi=dpi, bbox_inches=bbox_inches)

# Plot the error magnitudes compared to the total angle magnitude.
# Make a linear plot.
fig, ax = plt.subplots()
fig, ax = plot_vars(ax, ax.plot, total_angle_mag_flat_deg, attitudes_flat)
ax.set_xlabel('Total angle magnitude (deg)')
if show_plots:
    plt.show()
fig.savefig(file_path + "-total-angle-magnitude-linear.png",
            dpi=dpi, bbox_inches=bbox_inches)

# Make a semi-log plot.
fig, ax = plt.subplots()
fig, ax = plot_vars(ax, ax.semilogy, total_angle_mag_flat_deg, attitudes_flat)
ax.set_xlabel('Total angle magnitude (deg)')
if show_plots:
    plt.show()
fig.savefig(file_path + "-total-angle-magnitude-semilog.png",
            dpi=dpi, bbox_inches=bbox_inches)

# # OLD VERSIONS
# fig, ax = plt.subplots()

# ax.plot(angle_mag_flat_deg, dcm_mag_error.flatten(), 'o', label='Euler Angle DCM')
# ax.plot(angle_mag_flat_deg, qtr_mag_error.flatten(), 'x', label='Quaternion')
# ax.plot(angle_mag_flat_deg, mrp_mag_error.flatten(), '^', label='MRP')

# ax.set_xlabel('Angle magnitude (deg)')
# ax.set_ylabel('Error magnitude ($L_2$ norm of difference)')
# ax.legend()
# plt.show()

# # Make a semi-log plot.
# fig, ax = plt.subplots()

# ax.semilogy(angle_mag_flat_deg, dcm_mag_error.flatten(), 'o', label='Euler Angle DCM')
# ax.semilogy(angle_mag_flat_deg, qtr_mag_error.flatten(), 'x', label='Quaternion')
# ax.semilogy(angle_mag_flat_deg, mrp_mag_error.flatten(), '^', label='MRP')

# ax.set_xlabel('Angle magnitude (deg)')
# ax.set_ylabel('Error magnitude ($L_2$ norm of difference)')
# ax.legend()
# plt.show()

# # Plot the total angle magnitude.
# fig, ax = plt.subplots()

# ax.plot(total_angle_mag_flat_deg, dcm_mag_error.flatten(), 'o', label='Euler Angle DCM')
# ax.plot(total_angle_mag_flat_deg, qtr_mag_error.flatten(), 'x', label='Quaternion')
# ax.plot(total_angle_mag_flat_deg, mrp_mag_error.flatten(), '^', label='MRP')

# ax.set_xlabel('Total angle magnitude (deg)')
# ax.set_ylabel('Error magnitude ($L_2$ norm of difference)')
# ax.legend()
# plt.show()

# # Make a semi-log plot.
# fig, ax = plt.subplots()

# ax.semilogy(total_angle_mag_flat_deg, dcm_mag_error.flatten(), 'o', label='Euler Angle DCM')
# ax.semilogy(total_angle_mag_flat_deg, qtr_mag_error.flatten(), 'x', label='Quaternion')
# ax.semilogy(total_angle_mag_flat_deg, mrp_mag_error.flatten(), '^', label='MRP')

# ax.set_xlabel('Total angle magnitude (deg)')
# ax.set_ylabel('Error magnitude ($L_2$ norm of difference)')
# ax.legend()
# plt.show()
