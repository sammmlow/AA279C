# Purpose:
# Calculate the area of the outer spacecraft components
#
# Authors: Daniel Neamati and Sam Low

import numpy as np
from rom_utils import read_param_csv

# Read in the spacecraft rom geometry params from file as dictionary
geometry_param_file = "rom/spacecraft_rom_params.csv"
geo_params = read_param_csv(geometry_param_file)

# Abstract function to calculate the area of a circle
def circle_area(radius: float):
    """
    Calculate the area of a circle given the radius.
    """
    return np.pi * radius**2


# Manually specify the planar faces of the spacecraft

# Docking Port face
docking_port_area = circle_area(geo_params["rdockingport"])

# Hub face (annulus)
hub_face_area = circle_area(geo_params["rhub"]) - docking_port_area

# Solar panel faces
# x -> w
# y -> l
# z -> h
xw = geo_params["wsolarpanel"]
yl = geo_params["lsolarpanel"]
zh = geo_params["hsolarpanel"]
sp_x_area = yl * zh
sp_y_area = xw * zh
sp_z_area = xw * yl

# Antenna face
antenna_area = circle_area(geo_params["rantenna"])

# Store the areas in a dictionary
area_dict = {
    "hub": hub_face_area,
    "solar_panel_x": sp_x_area,
    "solar_panel_y": sp_y_area,
    "solar_panel_z": sp_z_area,
    "antenna": antenna_area,
    "docking_port": docking_port_area
}

# Print the areas
def print_area_dict(area_dict_: dict):
    """Helper function to print the areas from a dictionary."""
    for key_, value_ in area_dict_.items():
        print(f"{key_} area: {value_:.2f} m^2")

print("\n Planar Areas:")
print_area_dict(area_dict)

# Save the areas to a file
def save_area_dict(area_dict_: dict, output_file: str):
    """Helper function to save the areas to a file."""
    with open(output_file, "w") as f:
        for key_, value_ in area_dict_.items():
            f.write(f"{key_},{value_}\n")


area_output_file = "rom/area_planar_output.csv"
save_area_dict(area_dict, area_output_file)


############################

# Get the maximum projected area of the non planar surfaces

# Fuel Tank sphere -> circle
fuel_tank_proj_area = circle_area(geo_params["rfueltank"])

# Hub cylinder -> rectangle
hub_proj_area = 2 * geo_params["rhub"] * geo_params["hhub"]

# Antenna cylinder -> rectangle
antenna_proj_area = 2 * geo_params["rantenna"] * geo_params["hantenna"]

# Docking port cylinder -> rectangle
docking_port_proj_area = 2 * geo_params["rdockingport"] * \
                             geo_params["hdockingport"]


# Store the projected areas in a dictionary
proj_area_dict = {
    "fuel_tank": fuel_tank_proj_area,
    "hub": hub_proj_area,
    "antenna": antenna_proj_area,
    "docking_port": docking_port_proj_area
}

# Print the projected areas
print("\n Projected Areas:")
print_area_dict(proj_area_dict)

# Save the projected areas to a file
proj_area_output_file = "rom/area_proj_output.csv"
save_area_dict(proj_area_dict, proj_area_output_file)
