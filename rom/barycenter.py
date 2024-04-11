# Purpose:
# Calculate the barycenter of the outer spacecraft components
#
# Authors: Daniel Neamati and Sam Low

from rom_utils import read_param_csv, read_param_array_csv
import numpy as np

# Read in the spacecraft rom geometry params from file as dictionary
geometry_param_file = "rom/spacecraft_rom_params.csv"
geo_params = read_param_csv(geometry_param_file)

com_param_file = "rom/com_output.csv"
com_params = read_param_array_csv(com_param_file)

# For convenience, numpy arrays for the coordinate axes
x = np.array([1, 0, 0])
y = np.array([0, 1, 0])
z = np.array([0, 0, 1])

# We start with the planar faces of the spacecraft

# Hub face (annulus)
hub_face_vec = -x * geo_params["hhub"] / 2
hub_face_bary = com_params["hub"] + hub_face_vec

# Solar panel faces
xw = geo_params["wsolarpanel"]
yl = geo_params["lsolarpanel"]
zh = geo_params["hsolarpanel"]

sp_vecs = {
    "xplus": x * xw / 2,
    "xminus": -x * xw / 2,
    "yplus": y * yl / 2,
    "yminus": -y * yl / 2,
    "zplus": z * zh / 2,
    "zminus": -z * zh / 2
}
sp_barys_left = {face: com_params["sol left"]  +
                 vec for face, vec in sp_vecs.items()}
sp_barys_right = {face: com_params["sol right"] +
                  vec for face, vec in sp_vecs.items()}

# Antenna face
antenna_vec = z * geo_params["hantenna"] / 2
antenna_bary = com_params["ant"] + antenna_vec

# Docking port face
docking_port_vec = -x * geo_params["hdockingport"] / 2
docking_port_bary = com_params["dock"] + docking_port_vec

# Store the barycenters in a dictionary
planar_barys = {
    "hub": hub_face_bary
}
# add the solar panel barycenters
for side, sp_bary_side in zip(("left", "right"), (sp_barys_left, sp_barys_right)):
    for key, value in sp_bary_side.items():
        new_key = f"solar_panel_{side}_{key}"
        planar_barys[new_key] = value

# add the remaining barycenters
planar_barys["antenna"] = antenna_bary
planar_barys["docking_port"] = docking_port_bary

# Print the barycenters
def print_bary_dict(bary_dict_: dict, latex: bool = False):
    """Helper function to print the barycenters from a dictionary."""
    for key_, value_ in bary_dict_.items():
        if latex:
            start_str = f"{key_:25} barycenter: "
            vec_str = f"$( {value_[0]:.2f}, {value_[1]:.2f}, {value_[2]:.2f} )"
            print(start_str + vec_str + "~\si{\meter}$")
        else:
            print(f"{key_} barycenter: {value_}")

print("\n Planar Barycenters:")
print_bary_dict(planar_barys)
print()
print("\n Planar Barycenters (Latex):")
print_bary_dict(planar_barys, latex=True)

# Save the barycenters to a file
def save_bary_dict(bary_dict_: dict, output_file: str):
    """Helper function to save the barycenters to a file."""
    with open(output_file, "w") as f:
        for key_, value_ in bary_dict_.items():
            f.write(f"{key_},{value_}\n")

bary_output_file = "rom/barycenter_planar_output.csv"
save_bary_dict(planar_barys, bary_output_file)


# For the projected areas, we can also print the LaTeX formatted values
print("\n Projected Areas (Latex):")
print_bary_dict(com_params, latex=True)
