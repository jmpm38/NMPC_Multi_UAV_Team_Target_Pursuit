import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use('science')

# Load data from the text file
try:
    data = pd.read_csv('target_trajectory_3.txt', header=None, names=['x_position', 'y_position', 'x_velocity', 'y_velocity'])
except Exception as e:
    print("Error loading data:", e)
    raise

# Verify the data structure
print("First few rows of the data:")
print(data.head())

# Ensure that data columns are all floats for plotting
data = data.astype(float)

# Plotting x_position vs y_position
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(np.array(data['x_position']), np.array(data['y_position']), 'o-')  # Simplified plotting
plt.title('Position Trajectory')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.grid(True)
# plt.ylim(-53, 128)  # Example: limits for x-axis
# plt.xlim(-3, 203)  # Example: limits for y-axis
plt.ylim(-33, 33)  
plt.xlim(-33, 33)
# plt.ylim(-500, 500)  # Example: limits for x-axis
# plt.xlim(-50, 950)  # Example: limits for y-axis
# plt.xlim(-200, 600)  # Example: limits for x-axis
# plt.ylim(-600, 200)  # Example: limits for y-axis
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])


# Plotting x_velocity vs y_velocity
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
size = np.array(data['x_velocity']).size
linspace = np.linspace(0, size, size)#*0.2, size/0.2)
plt.plot(linspace, np.array(data['x_velocity']), label="$v_{x,T}$")  # Simplified plotting
plt.plot(linspace, np.array(data['y_velocity']), label="$v_{y,T}$")  # Simplified plotting
plt.legend()
plt.title('Velocity Trajectory')
plt.xlabel('Time')
plt.ylabel('Velocity ($m/s$)')
plt.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Show the plots
plt.tight_layout()
plt.show(block=False)
input()
