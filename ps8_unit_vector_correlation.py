# We have measurements of unit vectors in 3D, but their coordinates must be
# correlated through the unit vector constraint. 
# i.e., sqrt(x^2 + y^2 + z^2) = 1
# We look at the 2D case and try to find the covariance matrix between the 
# x and y coordinates of the unit vector.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

num_samples = 40000

np.random.seed(0)

# Plotting parameters
file_save_path = 'figures/ps8/PS8-UnitVecCorr'
dpi = 300
bbox_inches = 'tight'

#################################################

def fancy_plot_covariance(cov, show_plot=False, limit=1):
    """
    Make a heatmap of the covariance matrix with labelled entries.
    """ 
    fig, ax = plt.subplots(figsize=(4, 4))
    matax = ax.matshow(cov, cmap='coolwarm', vmin=-limit, vmax=limit)
    # Add colorbar
    cbar = plt.colorbar(matax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Covariance')

    # Add text labels
    for (i, j), val in np.ndenumerate(cov):
        ax.text(j, i, f'{val:.4f}', ha='center', va='center', color='black')

    ax.set_xticks([0, 1], labels=['x', 'y'])
    ax.set_yticks([0, 1], labels=['x', 'y'])

    plt.tight_layout()

    if show_plot:
        plt.show()
    return fig


def solve_ellipse_eigen(sigma_inv):
    """
    Solve the eigenvalues and eigenvectors. Return in order of highest to lowest
    :param sigma_inv: The inverse of the covariance matrix
    :return: the eigenvalues and eigenvectors
    """
    eig_vals, eig_vectors = np.linalg.eig(sigma_inv)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vectors = eig_vectors[:, idx]

    return eig_vals, eig_vectors


def solve_ellipse_params(eig_vals, eig_vectors, prob_conf):
    """
    Calculate the semi-major axis, semi-minor axis, and rotation angle from the eigen decomposition of the inverse of
    the covariance matrix
    :param eig_vals: The eigenvalues of the inverse of the covariance matrix
    :param eig_vectors: The eigenvectors of the inverse of the covariance matrix
    :param prob_conf: The probability for this confidence ellipse
    :return: semi-major axis (a), semi-minor axis (b), angle (theta)
    """

    # Numerator for semi axes
    prob_param = -2 * np.log(1 - prob_conf)
    # Assert that the above is positive to have real valued axes
    assert prob_param > 0
    # Calculate the axes
    b_semi_minor, a_semi_major = np.sqrt(prob_param / eig_vals)
    # Calculate the angle
    x_hat = np.array([1, 0])
    if eig_vectors[:, 1].shape[0] == 2:
        theta = np.arctan2(eig_vectors[1, 1], eig_vectors[0, 1])
    else:
        theta = np.arccos(x_hat @ eig_vectors[:, 1])

    return a_semi_major, b_semi_minor, theta


def plot_ellipse_updated(mu, sigma, p_conf, color, ax, lw=2):
    """
    Change the plotting function from HW 2 to work with this HW
    :param mu: The mean
    :param sigma: The covariance matrix
    :param p_conf: The probability
    :param color: The ellipse color
    :param ax: The axes
    :param lw: The line width
    :return: None
    """
    # print(sigma)
    sigma_inv = np.linalg.inv(sigma)
    eig_lambda, eig_vec = solve_ellipse_eigen(sigma_inv)
    a, b, t = solve_ellipse_params(eig_lambda, eig_vec, p_conf)

    # Make the ellipse
    ellipse = Ellipse(
        xy=mu, width=2 * a, height=2 * b, angle=np.rad2deg(t),
        facecolor=(*color, 0.2), edgecolor=(*color, 1),
        linewidth=lw, zorder=2)

    # Plot the ellipse
    ax.add_patch(ellipse)

    return ellipse


###############################################


# Generate n samples of 2D unit vectors uniformly across the unit circle
theta_samples = np.random.uniform(0, 2*np.pi, num_samples)
x_samples = np.cos(theta_samples)
y_samples = np.sin(theta_samples)

# Plot the samples
fig, ax = plt.subplots()
ax.scatter(x_samples, y_samples, alpha=0.01, marker='.', s=0.1)

box_lim = 1.2
ax.set_xlim(-box_lim, box_lim)
ax.set_ylim(-box_lim, box_lim)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel("x")
ax.set_ylabel("y")

fig.savefig(f"{file_save_path}_unit_vectors_uniform.png", dpi=dpi,
            bbox_inches=bbox_inches)

# Calculate the covariance matrix between the x and y coordinates
covariance_matrix = np.cov(x_samples, y_samples)

print("Covariance matrix: ")
print(covariance_matrix)

fig = fancy_plot_covariance(covariance_matrix)
fig.savefig(f"{file_save_path}_COV_uniform.png", dpi=dpi, bbox_inches=bbox_inches)

# Now suppose we have a 2D unit vector with x and y coordinates, but we
# have (small) noise added as a rotation.

u_ang_ref = [np.pi/3, 2 * np.pi/3, -np.pi/4]
u_vec_ref = [np.array([np.cos(t), np.sin(t)]) for t in u_ang_ref]

perturb_angle_std = np.deg2rad(5)
perturb_angle_samples = np.random.normal(0, perturb_angle_std, num_samples)

# Apply the perturbation to the reference unit vectors
u_ang_perturbed = [t + perturb_angle_samples for t in u_ang_ref]
u_vec_perturbed = [np.array([np.cos(t), np.sin(t)]) for t in u_ang_perturbed]
u_vec_perturbed = np.array(u_vec_perturbed)

# Calculate the covariance matrix between the x and y coordinates
# separately for the perturbed unit vectors

cov_matrices = [np.cov(u_vec_perturbed[i][0], u_vec_perturbed[i][1]
                       ) for i in range(3)]

print("Covariance matrices: ")
for i in range(3):
    print(f"Unit vector {i}: ")
    print(cov_matrices[i])
    fig_i = fancy_plot_covariance(cov_matrices[i], limit=0.01)
    fig_i.savefig(f"{file_save_path}_COV_perturbed_{i}.png", 
                  dpi=dpi, bbox_inches=bbox_inches)

# Plot the unit vectors and their perturbations
fig, ax = plt.subplots()

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# First plot the unit circle for reference
circle = plt.Circle((0, 0), 1, color='k', fill=False, linewidth=0.2)
ax.add_artist(circle)

for i in range(len(u_ang_ref)):
    ax.quiver(0, 0, u_vec_ref[i][0], u_vec_ref[i][1], 
              angles='xy', scale_units='xy', scale=1, 
              label=f"Unit vector {i + 1}", 
              color=default_colors[i])

# Reset colors
for i in range(len(u_ang_ref)):
    ax.scatter(u_vec_perturbed[i, 0, :], 
               u_vec_perturbed[i, 1, :],
               alpha=0.01,
               color=default_colors[i],)

# Plot the Gaussian ellipse approximation.
# Calculate the mean of the perturbed unit vectors
mean_vectors = np.mean(u_vec_perturbed, axis=-1)
print("Mean vectors: ")
print(mean_vectors)

# Plot the Gaussian ellipse approximation
# Reset colors
ax.set_prop_cycle(None)
for i in range(len(u_ang_ref)):
    plot_ellipse_updated(mean_vectors[i, :], cov_matrices[i], 0.95, 'k', ax)

box_lim = 1.2
ax.set_xlim(-box_lim, box_lim)
ax.set_ylim(-box_lim, box_lim)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.legend(loc='lower left')

fig.savefig(f"{file_save_path}_unit_vectors_distribution.png", dpi=dpi, 
            bbox_inches=bbox_inches)

