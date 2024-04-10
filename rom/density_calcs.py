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
hub_fracs = [0.1, 0.05, 0.0, 0.85]
solar_panel_fracs = [0.2, 0.2, 0.0, 0.6]
fuel_tank_fracs = [1.0, 0.0, 0.0, 0.0]
fuel_in_tank_fracs = [0.0, 0.0, 1.0, 0.0]
antenna_fracs = [0.15, 0.1, 0.0, 0.75]
docking_port_fracs = [0.05, 0.05, 0.0, 0.9]

part_fracs = [hub_fracs, solar_panel_fracs, fuel_tank_fracs,
              fuel_in_tank_fracs, antenna_fracs, docking_port_fracs]

# Check that the fractions sum to 1
for fracs in part_fracs:
    assert np.isclose(sum(fracs), 1.0), f"Fractions do not sum to 1. {fracs}"

# simple lambda function to calculate density
density = lambda rhos, fracs: np.dot(rhos, fracs)

densities = [density(rhos, fracs) for fracs in part_fracs]

# Print the results
names = ["Hub", "Solar Panel", "Fuel Tank",
         "Fuel in Tank", "Antenna", "Docking Port"]
for name_part, rho_part in zip(names, densities):
    print(f"{name_part:12} density: {rho_part:.2f} {units}")

# Output the densities to a csv file
with open("rom/density_output.csv", "w") as f:
    f.write("Part, Density (g/cm^3)\n")
    for name_part, rho_part in zip(names, densities):
        f.write(f"{name_part},{rho_part:.2f}\n")
