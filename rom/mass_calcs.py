# # Purpose:
# Calculate mass for each component
#
# Authors: Daniel Neamati and Sam Low

import numpy as np
from rom_utils import read_param_csv

# load in the density params from file as dictionary
density_param_file = "rom/density_output.csv"
density_params = read_param_csv(density_param_file)

part_names = density_params.keys()
# We have two solar panels!
solar_panel = "Solar Panel"
solar_panel_ind = 2
part_names_rom = list(part_names).copy()
part_names_rom.insert(solar_panel_ind, solar_panel)
# Remove the arms, which are not part of the ROM
part_names_rom = part_names_rom[:-1]
print("part_names_rom: ", part_names_rom)

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

# Note that we have two! solar panels
volumes = [hub_volume,
           solar_panel_volume, solar_panel_volume,
           fuel_tank_volume, fuel_in_tank_volume,
           antenna_volume, docking_port_volume]
total_volume = sum(volumes)

# Calculate the mass of each part
# Densities are in g/cm^3, volumes are in m^3
# Let's use kg/m^3 for the density units
# 1 g/cm^3 = 1000 kg/m^3
masses = [density_params[part_name] * 1000 * volume
          for part_name, volume in zip(part_names_rom, volumes)]
total_mass = sum(masses)

# Print the results
units = "kg"
for name_part, mass_part in zip(part_names_rom, masses):
    print(f"{name_part:12} mass: {mass_part:.2f} {units}")
print(f"Total mass: {total_mass:.2f} {units}")

# Print the results in LaTeX format
print("\nLaTeX format:")
latex_mass_units = "\si{\kilo\gram}"
latex_vol_units = "\si{\cubic\meter}"
for name_part, mass_part, vol_part in zip(part_names_rom, masses, volumes):
    print(f"{name_part} & {mass_part:.0f} {latex_mass_units} & {vol_part:.2f} {latex_vol_units} \\\\")
print("\\hline")
print(f"Total & {total_mass:.0f} {latex_mass_units} & {total_volume:.2f} {latex_vol_units} \\\\")

# Save the volumes and masses to file
vol_mass_output_file = "rom/vol_mass_output.csv"

# Remove the redundant solar panel
part_names_rom.remove(solar_panel)
volumes.pop(solar_panel_ind)
masses.pop(solar_panel_ind)

with open(vol_mass_output_file, "w") as f:
    f.write("Part, Volume (m^3), Mass (kg)\n")
    for name_part, volume, mass in zip(part_names_rom, volumes, masses):
        f.write(f"{name_part},{volume:.3f},{mass:.2f}\n")
