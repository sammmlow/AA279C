# # Purpose:
# Calculate mass for each component
#
# Authors: Daniel Neamati and Sam Low

import numpy as np
import csv

# abstract function to read a csv file and load in
# the keys as strings and the values as floats.
def read_param_csv(file_name: str):
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        # Ignore the first row of the csv file
        next(reader)

        output_dict = {}

        for row in reader:
            output_dict[row[0]] = float(row[1])

    return output_dict

# load in the density params from file as dictionary
density_param_file = "rom/density_output.csv"
density_params = read_param_csv(density_param_file)

part_names = density_params.keys()

# load in the spacecraft geometry rom params from file as dictionary
geometry_param_file = "rom/spacecraft_rom_params.csv"
geometry_params = read_param_csv(geometry_param_file)

# abstract volume equations
cylinder_volume = lambda r, h: np.pi * (r**2) * h
cuboid_volume = lambda w, l, h: w * l * h
sphere_volume = lambda r: 4/3 * np.pi * (r**3)
hollow_sphere_volume = lambda r, t: 4 * np.pi * (r**2) * t

# assign the volumes for the parts
hub_volume = cylinder_volume(
    geometry_params["rhub"], geometry_params["hhub"]
)
solar_panel_volume = cuboid_volume(
    geometry_params["wsolarpanel"],
    geometry_params["lsolarpanel"],
    geometry_params["hsolarpanel"]
)
fuel_tank_volume = hollow_sphere_volume(
    geometry_params["rfueltank"],
    geometry_params["tfueltank"]
)
fuel_in_tank_volume = sphere_volume(
    geometry_params["rfueltank"]
)
antenna_volume = cylinder_volume(
    geometry_params["rantenna"],
    geometry_params["hantenna"]
)
docking_port_volume = cylinder_volume(
    geometry_params["rdockingport"],
    geometry_params["hdockingport"]
)

volumes = [hub_volume, solar_panel_volume, fuel_tank_volume,
           fuel_in_tank_volume, antenna_volume, docking_port_volume]

# Calculate the mass of each part
# Densities are in g/cm^3, volumes are in m^3
# Let's use kg/m^3 for the density units
# 1 g/cm^3 = 1000 kg/m^3
masses = [density_params[part_name] * 1000 * volume
          for part_name, volume in zip(part_names, volumes)]

# Print the results
units = "kg"
for name_part, mass_part in zip(part_names, masses):
    print(f"{name_part:12} mass: {mass_part:.2f} {units}")


# Save the volumes and masses to file
mass_output_file = "rom/vol_mass_output.csv"
with open(mass_output_file, "w") as f:
    f.write("Part, Volume (m^3), Mass (kg)\n")
    for name_part, volume, mass in zip(part_names, volumes, masses):
        f.write(f"{name_part},{volume:.3f},{mass:.2f}\n")
