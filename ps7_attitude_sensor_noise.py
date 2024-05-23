"""
Given a attitude sensor (i.e., direction vectors), with noise in terms of the 
deviation angle, we want to generate the sampled attitude sensor data (true
+ noise)
"""

import numpy as np
import matplotlib.pyplot as plt

import source.unit_vec_gaussian as uvg
import source.attitudes as att

# Plotting parameters
file_path = "figures/ps7/PS7-SensorNoise-"
show_plots = False
dpi = 200
bbox_inches = "tight"

# Monte Carlo simulation parameters
n_samples = 1000
n_bins = 50
np.random.seed(279)

# Gyro drift
delta_t = 300 # seconds
delta_t_hrs = delta_t / 3600 # hours
gyro_drift_per_hr = 0.0018 # degrees per hour
gyro_drift = gyro_drift_per_hr * delta_t_hrs
print("Gyro drift: ", gyro_drift)

# Values are in degrees
sensors = {
    "Sun Sensor": { "mean_angle": 0.5, "std_dev_angle": 2.5 },
    "Magnetometer": { "mean_angle": 1, "std_dev_angle": 2.5 },
    "Gyroscope": { "mean_angle": 0.2, "std_dev_angle": gyro_drift },
    "Star Tracker": { "mean_angle": 0.0025, "std_dev_angle": 0.01 }
}

# Now we sample the sensors, determine the angle induced by the noise, and
# generate the noisy sensor data.
test_vector = np.ones(3) / np.sqrt(3)

all_angles = np.zeros((len(sensors), n_samples))

for sensor_num, (sensor_name, sensor_params) in enumerate(sensors.items()):
    print("Sensor: ", sensor_name)
    print("Mean angle: ", sensor_params["mean_angle"])
    print("Standard deviation of angle: ", sensor_params["std_dev_angle"])
    print()
    
    uvg_sensor = uvg.UnitVecGaussian(
        mean_angle = sensor_params["mean_angle"] * np.pi / 180,
        std_dev_angle = sensor_params["std_dev_angle"] * np.pi / 180
    )
    
    samples = uvg_sensor.sample(n_samples)
    
    for i in range(n_samples):
        sample = samples[i]

        # Create the quaternion and apply the rotation
        qtr_curr = att.QTR( qtr = sample )
        qtr_curr.conventionalize()
        vec_new = qtr_curr.apply(test_vector)

        # Calculate the angle between the true and noisy vectors
        vec_dot = np.dot(
            test_vector / np.linalg.norm(test_vector), 
            vec_new / np.linalg.norm(vec_new)
            )
        vec_angle = np.arccos(vec_dot)

        all_angles[sensor_num, i] = np.degrees(vec_angle)


# Plot the results

for sensor_num, (sensor_name, sensor_params) in enumerate(sensors.items()):
    fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

    # Top plot is the histogram of the angles
    axes[0].hist(all_angles[sensor_num], bins = n_bins, label="Sampled Noise")
    axes[0].axvline(sensor_params["mean_angle"], color = "k", linestyle = "--",
                label="Prescribed Mean Angle")
    axes[0].axvline(sensor_params["mean_angle"] + sensor_params["std_dev_angle"], 
                color = "r", linestyle = "--",
                label="Mean Angle + Std Dev")
    
    if sensor_params["mean_angle"] - sensor_params["std_dev_angle"] > 0:
        axes[0].axvline(sensor_params["mean_angle"] - sensor_params["std_dev_angle"],
                    color = "r", linestyle = "--",
                    label="Mean Angle - Std Dev")
        
    axes[0].legend()
    axes[0].set_title(sensor_name)
    # axes[0].set_xlabel("Angle Induced by Noise (degrees)")
    axes[0].set_ylabel("Frequency")

    # Bottom plot is the box plot of the angles (oriented horizontally)
    # Thin plot as not to take up too much space
    axes[1].boxplot(all_angles[sensor_num], vert = False)
    axes[1].set_xlabel("Angle Induced by Noise (degrees)")
    axes[1].set_yticks([])

    plt.tight_layout()

    if show_plots:
        plt.show()

    fig.savefig(file_path + sensor_name + "_hist_box.png", 
                dpi = dpi, bbox_inches = bbox_inches)
