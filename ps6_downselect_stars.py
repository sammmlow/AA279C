# Down select the stars in teh HIPPARCOS catalog for simplicity

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in the HIPPARCOS catalog
hipparcos = pd.read_csv('hipparcos_star_catalog_solutions.txt', sep=" ", header=None)

print(hipparcos)

select_num = 50

# Down select the stars
stars_selector = np.arange(0, len(hipparcos), len(hipparcos)//select_num)[:select_num]
hipparcos_selected = hipparcos.iloc[stars_selector]

print(hipparcos_selected)

# Save the down selected stars
hipparcos_selected.to_csv(f'hipparcos_star_catalog_solutions_downselected_{select_num}.txt', 
                          sep=" ", header=None, index=False)

# Plot the xyz positions of the stars
# For the full case and the down selected case

catalogs = [hipparcos, hipparcos_selected]
catalog_names = ['HIPPARCOS', f'HIPPARCOS_Downselected_{select_num}']
sizes = [1, 10]

for stars, name, size in zip(catalogs, catalog_names, sizes):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(stars[0], stars[1], stars[2], s=size)

    # Plot the 2-Sphere for reference
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.1)

    # Use equal aspect ratio for all axes
    ax.set_box_aspect([1,1,1])

    # Set the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    # Save the figure
    fig.savefig(f'figures/ps6/PS6-Stars-{name}.png', dpi=200, bbox_inches='tight')
