import numpy as np

# Ellipse parameters
a = 5  # semi-major axis (half the range in x)
b = 12.5  # semi-minor axis (half the range in y)
final_velocity_magnitude = 3 # final target speed in m/s
dt = 0.2  # time step in seconds
total_time = 300  # total duration in seconds
t = np.arange(0, total_time, dt)  # time array

# Starting position offset
start_x = 0  # initial x position
start_y = 5  # initial y position

# Calculate the maximum angular speed needed for the target final speed
perimeter_estimate = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))  # Ramanujan approximation for ellipse perimeter
max_angular_speed = final_velocity_magnitude / perimeter_estimate * 2 * np.pi  # max angular speed to reach the final speed

# Initialize lists to store trajectory data
x_list, y_list, vx_list, vy_list = [], [], [], []

for time in t:
    # Angular speed linearly increases from 0 to max_angular_speed over time
    angular_speed = max_angular_speed * (time / total_time)
    theta = 0.5 * angular_speed * time  # Integrating angular acceleration to find theta
    # Compute the position and velocity on the ellipse with starting offset
    x = start_x + a * np.cos(theta)
    y = start_y + b * np.sin(theta)
    vx = -a * angular_speed * np.sin(theta)
    vy = b * angular_speed * np.cos(theta)
    
    x_list.append(x)
    y_list.append(y)
    vx_list.append(vx)
    vy_list.append(vy)

# Save data to a txt file
with open("target_trajectory.txt", "w") as file:
    for x, y, vx, vy in zip(x_list, y_list, vx_list, vy_list):
        file.write(f"{x:.4f},{y:.4f},{vx:.4f},{vy:.4f}\n")

print("Trajectory data saved to 'accelerating_ellipse_trajectory_with_start.txt'")
