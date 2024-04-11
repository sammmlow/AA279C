# Purpose:
# Calculate the Moment of Inertia (MOI) Tensor of the whole spacecraft assembly
#
# Authors: Daniel Neamati and Sam Low

import numpy as np

from rom_utils import read_param_array_csv
from moi_utils import moi_tensor_parallel_axis

# Read in the moment of inertia (MOI) params from file as dictionary
moi_parts_param_file = "rom/moi_part_output.csv"
moi_parts = read_param_array_csv(moi_parts_param_file)

# Read in the mass params from file as dictionary
mass_param_file = "rom/vol_mass_output.csv"
mass_parts = read_param_array_csv(mass_param_file)

# Read in the center of mass (com) params from file as dictionary
com_param_file = "rom/com_output.csv"
com_vecs = read_param_array_csv(com_param_file)

# Print the parts
print(f"Parts: {com_vecs.keys()}")

# Match the moi_parts and com_vecs keys
part_keys = {
    'hub': ("Hub",),
    'sol left': ("Solar Panel",),
    'sol right': ("Solar Panel",),
    'tank': ("Fuel Tank", "Fuel in Tank"),
    'ant': ("Antenna",),
    'dock': ("Docking Port",),
}

# Check the keys of part_keys match the keys of com_vecs
assert set(part_keys.keys()) == set(com_vecs.keys())

# Calculate the MOI at the COM of the assembly
moi_parts_at_assembly_com = {}
for part_key, part_key_tuple in part_keys.items():

    moi_part_vec = np.zeros(3)
    mass_part = 0

    for moi_mass_key in part_key_tuple:
        # Get the MOI of the part
        moi_part_vec += moi_parts[moi_mass_key]
        # Get the mass of the part
        mass_part += mass_parts[moi_mass_key][1]

    # Conver the moi_part_vec to a MOI tensor
    moi_part_tensor = np.diag(moi_part_vec)

    # Get the COM of the part
    com_part = com_vecs[part_key]

    # Print parameters for debugging
    print(f"\nPart: {part_key}")
    print("MOI")
    print(moi_part_tensor)
    print(f"Mass: {mass_part}")
    print(f"COM: {com_part}")

    # Calculate the MOI at the assembly COM
    moi_parts_at_assembly_com[part_key] = \
        moi_tensor_parallel_axis(moi_part_tensor, mass_part, com_part)

print("\n\n\n###############################################\n\n\n")

# Print the MOI at the assembly COM
for part_key, moi_part in moi_parts_at_assembly_com.items():
    print(f"MOI of {part_key} at assembly COM:")
    print(moi_part)

# Calculate the MOI of the assembly
moi_assembly = np.zeros((3, 3))
for moi_part in moi_parts_at_assembly_com.values():
    moi_assembly += moi_part

print("\n\n\n###############################################\n\n\n")

print("MOI of the assembly:")
print(moi_assembly)

# Plot without scientific notation
np.set_printoptions(suppress=True)
print()
print(moi_assembly)

# Print in LaTeX format
print()
for row in moi_assembly:
    print(" & ".join([f"{val:.3f}" for val in row]) + " \\\\")

# Save the MOI of the assembly to a file
moi_assembly_param_file = "rom/moi_assembly_output.csv"
np.savetxt(moi_assembly_param_file, moi_assembly, delimiter=",")


# Calculate the principal axes and principal moments of inertia
eigenvalues, eigenvectors = np.linalg.eig(moi_assembly)

print("\n###############################################\n")

print("Principal axes:")
for eig_ind, eigenvector in enumerate(eigenvectors):
    print(eig_ind, eigenvector)
    print(np.linalg.norm(eigenvector))
    print(f"{eigenvector[0]:.5f} & {eigenvector[1]:.5f} & {eigenvector[2]:.5f}")

print("\nPrincipal moments of inertia:")
for eigenvalue in eigenvalues:
    print(f"{eigenvalue:.3f}")
