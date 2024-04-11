# Purpose:
# Calculate the center of mass of the spacecraft
#
# Authors: Daniel Neamati and Sam Low

from rom_utils import read_param_csv

# Read in the mass params from file as dictionary
vol_mass_param_file = "rom/vol_mass_output.csv"
vol_mass_params = read_param_csv(vol_mass_param_file)

part_names = vol_mass_params.keys()



