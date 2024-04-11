# Purpose:
# Calculate moments of inertia for each component
#
# Authors: Daniel Neamati and Sam Low

import numpy as np

from moi_utils import moi_cuboid, moi_cylinder, \
    moi_solid_sphere, moi_hollow_sphere
from rom_utils import read_param_csv

# load in the mass params from file as dictionary
vol_mass_param_file = "rom/vol_mass_output.csv"
vol_mass_params = read_param_csv(vol_mass_param_file)

part_names = vol_mass_params.keys()

# load in the spacecraft geometry rom params from file as dictionary
geometry_param_file = "rom/spacecraft_rom_params.csv"
geometry_params = read_param_csv(geometry_param_file)

# Calculate the moments of inertia for each part about its center of mass
hub_moi = moi_cylinder(
    vol_mass_params["Hub"],
    geometry_params["rhub"], geometry_params["hhub"],
    axis=0
)
solar_panel_moi = moi_cuboid(
    vol_mass_params["Solar Panel"],
    geometry_params["wsolarpanel"],
    geometry_params["lsolarpanel"],
    geometry_params["hsolarpanel"]
)
fuel_tank_moi = moi_hollow_sphere(
    vol_mass_params["Fuel Tank"],
    geometry_params["rfueltank"]
)
fuel_in_tank_moi = moi_solid_sphere(
    vol_mass_params["Fuel in Tank"],
    geometry_params["rfueltank"]
)
antenna_moi = moi_cylinder(
    vol_mass_params["Antenna"],
    geometry_params["rantenna"], geometry_params["hantenna"],
    axis=2
)
docking_port_moi = moi_cylinder(
    vol_mass_params["Docking Port"],
    geometry_params["rdockingport"], geometry_params["hdockingport"],
    axis=0
)

# All the MOIs
mois = [hub_moi, solar_panel_moi, fuel_tank_moi,
        fuel_in_tank_moi, antenna_moi, docking_port_moi]

# Print the results
for part_name, moi in zip(part_names, mois):
    print(f"{part_name}:")
    print(moi)
    print("\n")

# Save the MOIs to a CSV.
# Since all are diagonal matrices, we can save just the diagonal elements.
moi_diag = [np.diag(moi) for moi in mois]

moi_save_filename = "rom/moi_part_output.csv"
with open(moi_save_filename, "w") as f:
    f.write("Part,MOI1,MOI2,MOI3\n")
    for part_name, diag in zip(part_names, moi_diag):
        f.write(f"{part_name},{diag[0]},{diag[1]},{diag[2]}\n")
