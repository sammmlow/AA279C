# Purpose:
# Calculate relative vectors between component CoM to full assembly CoM
# Note that the coordinate frame expressed in here is our own definition where
# X is along the axis of lowest rotational inertia, Z is along the largest
#
# Authors: Daniel Neamati and Sam Low

import numpy as np

# ========================================================================
# ========================================================================
# ALL CALCULATIONS BELOW ARE DONE FOR THE REDUCED ORDER MODEL

# Assembly CoM (w.r.t. some arbitrary origin in Solidworks)
full_assembly_com = np.array([-0.23982, -2.241, 0.449292])

# Component-wise CoM (w.r.t. some assembly CoM)
hub_com            = np.array([ 2.240996, -1.665585,  0.446156])
solar_left_com     = np.array([-0.759004, -1.665585,  0.451156])
solar_right_com    = np.array([ 5.240996, -1.665585,  0.451156])
fuel_tank_full_com = np.array([ 2.240995,  0.058515,  0.446156])
antenna_com        = np.array([ 2.243550, -1.674850,  1.458610])
docking_port_com   = np.array([ 2.240990, -2.365585,  0.446156])

hub_com            = np.array([-1.665585, -2.240996, 0.446156])
solar_left_com     = np.array([-1.665585,  0.759004, 0.451156])
solar_right_com    = np.array([-1.665585, -5.240996, 0.451156])
fuel_tank_full_com = np.array([ 0.058515, -2.240995, 0.446156])
antenna_com        = np.array([-1.674850, -2.243550, 1.458610])
docking_port_com   = np.array([-2.365585, -2.240990, 0.446156])

part_names = ['hub', 'solar left', 'solar right', 'full tank', 'antenna', 
              'docking port']

# Compute relative component CoM w.r.t. full assembly CoM
hub_com_rel            = -full_assembly_com + hub_com
solar_left_com_rel     = -full_assembly_com + solar_left_com
solar_right_com_rel    = -full_assembly_com + solar_right_com
fuel_tank_full_com_rel = -full_assembly_com + fuel_tank_full_com
antenna_com_rel        = -full_assembly_com + antenna_com
docking_port_com_rel   = -full_assembly_com + docking_port_com

# Print the results
print("Component-wise CoM w.r.t. Assembly CoM:")
print("CoM of hub = ",            -full_assembly_com + hub_com)
print("CoM of solar left = ",     -full_assembly_com + solar_left_com)
print("CoM of solar right = ",    -full_assembly_com + solar_right_com)
print("CoM of full fuel tank = ", -full_assembly_com + fuel_tank_full_com)
print("CoM of antenna = ",        -full_assembly_com + antenna_com)
print("CoM of docking port = ",   -full_assembly_com + docking_port_com)

# List of parts
list_of_parts_com = [hub_com_rel,
                     solar_left_com_rel,
                     solar_right_com_rel,
                     fuel_tank_full_com_rel,
                     antenna_com_rel,
                     docking_port_com_rel]
list_of_parts_name = ['hub', 'sol left', 'sol right', 'tank', 'ant', 'dock']

# Save the relative CoMs to a CSV.
com_save_filename = "com_output.csv"
with open(com_save_filename, "w") as f:
    f.write("Part,comX,comY,comZ\n")
    for part_name, part_com in zip(list_of_parts_name, list_of_parts_com):
        f.write(f"{part_name},{part_com[0]},{part_com[1]},{part_com[2]}\n")