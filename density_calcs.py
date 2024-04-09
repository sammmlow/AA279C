# Purpose:
# Calculate density mixtures as linear combination of material components
#
# Authors: Daniel Neamati and Sam Low

import numpy as np

# MATERIAL DENSITIES
rho_ti = 4.43 # g/cm^3, Ti-6Al-4V
rho_si = 2.33 # g/cm^3, Silicon
rho_xe = 0.5 # g/cm^3, Xenon
rho_empty = 0.0 # g/cm^3, Empty space

units = "g/cm^3"

rhos = [rho_ti, rho_si, rho_xe, rho_empty]

# VOLUMETRIC FRACTIONS
hub_fracs = [0.7, 0.2, 0.0, 0.1]
solar_panel_fracs = [0.25, 0.75, 0.0, 0.0]
fuel_tank_fracs = [1.0, 0.0, 0.0, 0.0]
fuel_in_tank_fracs = [0.0, 0.0, 1.0, 0.0]
antenna_fracs = [0.25, 0.75, 0.0, 0.0]
docking_port_fracs = [0.9, 0.1, 0.0, 0.0]

part_fracs = [hub_fracs, solar_panel_fracs, fuel_tank_fracs,
              fuel_in_tank_fracs, antenna_fracs, docking_port_fracs]

# simple lambda function to calculate density
density = lambda rhos, fracs: np.dot(rhos, fracs)

densities = [density(rhos, fracs) for fracs in part_fracs]

# Print the results
names = ["Hub", "Solar Panel", "Fuel Tank",
         "Fuel in Tank", "Antenna", "Docking Port"]
for name_part, rho_part in zip(names, densities):
    print(f"{name_part:12} density: {rho_part:.2f} {units}")

# Output the densities to a csv file
with open("density_output.csv", "w") as f:
    f.write("Part, Density (g/cm^3)\n")
    for name_part, rho_part in zip(names, densities):
        f.write(f"{name_part},{rho_part:.2f}\n")
