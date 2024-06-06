###########################
# Actuator mounting models

import numpy as np
import matplotlib.pyplot as plt

file_path = "figures/ps9/PS9-ACTUATOR-OPTS-"
dpi=200
bbox_inches='tight'

# First just check that the mounting matrices have reasonable pseudo-inverses
a_rw = 1/np.sqrt(3) * np.array(
    [[ 1,  1,  1],
     [ 1,  1, -1],
     [ 1, -1,  1],
     [ 1, -1, -1],
     [-1,  1,  1],
     [-1,  1, -1],
     [-1, -1,  1],
     [-1, -1, -1]]
).T

a_rw_pinv = np.linalg.pinv(a_rw)

print(a_rw)
print(a_rw.shape)
print(a_rw_pinv)
print(a_rw_pinv.shape)
print(np.linalg.det(a_rw @ a_rw_pinv))
print(np.linalg.det(a_rw_pinv @ a_rw))
print(np.linalg.norm(a_rw @ a_rw_pinv @ a_rw - a_rw))

print(np.sqrt(3) / 8)
print(8 / np.sqrt(3) * a_rw_pinv)

# Get the condition number of the mounting matrix
print(np.linalg.cond(a_rw))
print(np.linalg.cond(a_rw_pinv))


# Make a 3D plot of the reaction wheels axes from the mounting matrix
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.subplots_adjust(right=0.7)

for r_ind, rw in enumerate(a_rw.T):
    # Axial line
    ax.plot([rw[0], 2*rw[0]], [rw[1], 2*rw[1]], [rw[2], 2*rw[2]], 'o-',
            label=f"Wheel {r_ind+1} Axis")

# Plot the unit cube
# Technically... some of the lines are redundant, but it's easier to just plot 
# them all
labelled_cube = False
normalizer = 1/np.sqrt(3)
for xsign in [-1 * normalizer, 1 * normalizer]:
    for ysign in [-1 * normalizer, 1 * normalizer]:
        for zsign in [-1 * normalizer, 1 * normalizer]:
            ax.plot([xsign, xsign], [ysign, ysign], [zsign, -zsign], 'k:')
            ax.plot([xsign, xsign], [ysign, -ysign], [zsign, zsign], 'k:')
            ax.plot([xsign, -xsign], [ysign, ysign], [zsign, zsign], 'k:',
                    label=f"Cube" if not labelled_cube else None)
            labelled_cube = True

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_box_aspect([1.0, 1.0, 1.0])
plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
# plt.title("Reaction wheel axes")
plt.tight_layout()
plt.savefig(file_path + "rw_axes.png", dpi=dpi, bbox_inches=bbox_inches)
# plt.show()


###########################
# Now check the thruster mounting matrix

thruster_x = -2.0
thruster_pos_angle = np.deg2rad(30.0)
thruster_angle = np.deg2rad(30.0)
thruster_az_angle = np.deg2rad(45.0)
hub_rad = 0.5

cp = np.cos(thruster_pos_angle)
sp = np.sin(thruster_pos_angle)

cd = np.cos(thruster_angle)
sd = np.sin(thruster_angle)

ca = np.cos(thruster_az_angle)
sa = np.sin(thruster_az_angle)

thruster_positions = np.array(
    [[thruster_x, hub_rad * cp, hub_rad * sp],
     [thruster_x, hub_rad * cp, -hub_rad * sp],
     [thruster_x, -hub_rad * cp, hub_rad * sp],
     [thruster_x, -hub_rad * cp, -hub_rad * sp]]
)
thruster_angles = np.array(
    [[-cd, sd*ca, sd*sa],
     [-cd, sd*ca, -sd*sa],
     [-cd, -sd*ca, sd*sa],
     [-cd, -sd*ca, -sd*sa]]
)

mounting_vectors = []
for i in range(4):
    # Check that the angle vectors are unit vectors
    print(np.linalg.norm(thruster_angles[i]))
    assert np.isclose(np.linalg.norm(thruster_angles[i]), 1.0)
    mounting_vectors.append(np.cross(thruster_positions[i], thruster_angles[i]))

a_thruster = np.array(mounting_vectors).T
a_thruster_pinv = np.linalg.pinv(a_thruster)

print("Thruster matrix")
print("Thruster positions")
print(thruster_positions)
print("Thruster angles")
print(thruster_angles)
print("Mounting vectors")
print(a_thruster)
print(a_thruster.shape)
print("Pseudo-inverse")
print(a_thruster_pinv)
print(a_thruster_pinv.shape)
print("LaTeX format")
for idx in a_thruster_pinv:
    print(" & ".join([f"{val:.3f}" for val in idx]) + r" \\")
print("Check pseudo-inverse")
print(a_thruster @ a_thruster_pinv)
print(a_thruster_pinv @ a_thruster)
print("Determinants and norms")
print(np.linalg.det(a_thruster @ a_thruster_pinv))
print(np.linalg.det(a_thruster_pinv @ a_thruster))
print(np.linalg.norm(a_thruster @ a_thruster_pinv @ a_thruster - a_thruster))
print("Condition numbers")
print(np.linalg.cond(a_thruster))
print(np.linalg.cond(a_thruster_pinv))

# Make a 3D plot of the thruster positions and axes from the mounting matrix
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.subplots_adjust(right=0.7)

labelled_loa = False
loa_factor = 2.5
for t_ind, (thruster_pos, thruster_angle) in enumerate(zip(thruster_positions, thruster_angles)):

    # Plot the line of action
    ax.plot([thruster_pos[0] - loa_factor * thruster_angle[0], thruster_pos[0]],
            [thruster_pos[1] - loa_factor * thruster_angle[1], thruster_pos[1]],
            [thruster_pos[2] - loa_factor * thruster_angle[2], thruster_pos[2]],
            'k:',
            label=f"Line of action" if not labelled_loa else None)
    labelled_loa = True
    # Thruster position and axis
    ax.plot([thruster_pos[0], thruster_pos[0] + thruster_angle[0]],
            [thruster_pos[1], thruster_pos[1] + thruster_angle[1]],
            [thruster_pos[2], thruster_pos[2] + thruster_angle[2]], 'o-',
            label=f"Thruster {t_ind+1}")


# Origin
ax.plot([0], [0], [0], 'kx', label='Center of mass')

# Hub circle
theta = np.linspace(0, 2*np.pi, 100)
x = thruster_x * np.ones_like(theta)
y = hub_rad * np.cos(theta)
z = hub_rad * np.sin(theta)
ax.plot(x, y, z, 'r--', label='Hub circle', zorder=-1)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')

x_lims = ax.set_xlim([-3, 0.5])
y_lims = ax.set_ylim([-1, 1])
z_lims = ax.set_zlim([-1, 1])

x_length = x_lims[1] - x_lims[0]
y_length = y_lims[1] - y_lims[0]
z_length = z_lims[1] - z_lims[0]

ax.set_yticks(np.linspace(y_lims[0], y_lims[1], 5))
ax.set_zticks(np.linspace(z_lims[0], z_lims[1], 5))

ax.view_init(elev=30, azim=115)
# ax.set_box_aspect([1.0, 1.0, 1.0])
ax.set_box_aspect([x_length, y_length, z_length])
plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
# plt.title("Thruster positions and axes")
plt.tight_layout()
fig.savefig(file_path + "thruster_axes.png", dpi=dpi, bbox_inches=bbox_inches)
# fig.show()


###########################
# Plot the 2D projections of the thruster positions and axes on the yz plane

fig, ax = plt.subplots()

labelled_loa = False
loa_factor = 2.5
for t_ind, (thruster_pos, thruster_angle) in enumerate(zip(thruster_positions, thruster_angles)):

    # Plot the line of action
    ax.plot([thruster_pos[1] - loa_factor * thruster_angle[1], thruster_pos[1]],
            [thruster_pos[2] - loa_factor * thruster_angle[2], thruster_pos[2]],
            'k:',
            label=f"Line of action" if not labelled_loa else None)
    labelled_loa = True
    # Thruster position and axis
    ax.plot([thruster_pos[1], thruster_pos[1] + thruster_angle[1]],
            [thruster_pos[2], thruster_pos[2] + thruster_angle[2]], 'o-',
            label=f"Thruster {t_ind+1}")


# Origin
ax.plot([0], [0], 'kx', label='Center of mass')

# Hub circle
theta = np.linspace(0, 2*np.pi, 100)
y = hub_rad * np.cos(theta)
z = hub_rad * np.sin(theta)
ax.plot(y, z, 'r--', label='Hub circle', zorder=-1)

ax.set_xlabel('Y [m]')
ax.set_ylabel('Z [m]')

ax.set_aspect('equal')

ax.set_xticks(np.linspace(y_lims[0], y_lims[1], 5))
ax.set_yticks(np.linspace(z_lims[0], z_lims[1], 5))

ax.invert_xaxis()
plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
# plt.title("Thruster positions and axes")
plt.tight_layout()
fig.savefig(file_path + "thruster_axes_2d.png", dpi=dpi, bbox_inches=bbox_inches)
# fig.show()
