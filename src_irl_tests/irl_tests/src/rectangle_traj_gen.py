import numpy as np

# Parameters
initial_position = np.array([0.0, 5.0], dtype=float)  # Starting position (x, y)
target_speed = 1.5  # Target speed (m/s) on straight segments
corner_speed_factor = 0.7  # Speed factor for corners
time_step = 0.2  # Time step (s)
num_points = 500  # Total points to simulate

# Rounded rectangle bounds and radii
x_min, x_max = 0, 10
y_min, y_max = 5, 30
corner_radius = 2  # Radius for rounded corners (m)

# Initialize variables
trajectory_data = []
speed = 0.0
acceleration = 0.05  # Acceleration rate to target speed

# Functions for straight and curved movements
def move_straight(start_pos, direction, dist_remaining, current_speed, duration):
    """Generates straight-line points based on current position and speed."""
    points = []
    dist_traveled = 0
    pos = start_pos.copy()
    while dist_traveled < dist_remaining:
        pos += direction * current_speed * duration
        vx, vy = direction * current_speed
        points.append((pos[0], pos[1], vx, vy))
        dist_traveled += current_speed * duration
    return points

def move_corner(center, angle_start, angle_end, radius, corner_speed):
    """Generates points for a rounded corner based on arc length and adjusted speed."""
    points = []
    angle_range = np.linspace(angle_start, angle_end, int(radius / (corner_speed * time_step)) + 1)
    for angle in angle_range:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        vx = -np.sin(angle) * corner_speed
        vy = np.cos(angle) * corner_speed
        points.append((x, y, vx, vy))
    return points

# Generate trajectory around the rectangle with rounded corners
# Define each segment and corner
segments_and_corners = [
    (move_straight, (np.array([0.0, 5.0]), np.array([1.0, 0.0]), x_max - x_min - 1*corner_radius, target_speed, time_step)),
    (move_corner, ([x_max - corner_radius, y_min + corner_radius], -np.pi / 2, 0, corner_radius, target_speed * corner_speed_factor)),
    (move_straight, (np.array([x_max, y_min + 2.0]), np.array([0.0, 1.0]), y_max - y_min - 2*corner_radius, target_speed, time_step)),
    (move_corner, ([x_max - corner_radius, y_max - corner_radius], 0, np.pi / 2, corner_radius, target_speed * corner_speed_factor)),
    (move_straight, (np.array([x_max - 2.0, y_max]), np.array([-1.0, 0.0]), x_max - x_min - 2*corner_radius, target_speed, time_step)),
    (move_corner, ([x_min + corner_radius, y_max - corner_radius], np.pi / 2, np.pi, corner_radius, target_speed * corner_speed_factor)),
    (move_straight, (np.array([x_min, y_max - 2.0]), np.array([0.0, -1.0]), y_max - y_min - 2*corner_radius, target_speed, time_step)),
    (move_corner, ([x_min + corner_radius, y_min + corner_radius], np.pi, 3 * np.pi / 2, corner_radius, target_speed * corner_speed_factor))
]

# Collect all points from each segment and corner
for func, args in segments_and_corners:
    trajectory_data.extend(func(*args))

# Write the trajectory data to a .txt file
output_path = 'target_trajectory.txt'
with open(output_path, 'w') as file:
    for data_point in trajectory_data:
        file.write(f"{data_point[0]:.2f},{data_point[1]:.2f},{data_point[2]:.2f},{data_point[3]:.2f}\n")

print("Trajectory data has been generated and saved as 'trajectory_data.txt'.")
