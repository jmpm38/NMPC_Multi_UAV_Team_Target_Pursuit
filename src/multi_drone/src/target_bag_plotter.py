from bagpy import bagreader
import csv
import numpy as np
import math
import target_graphs2 as graphs
import scipy.interpolate as interp
import yaml

def yaml_handling():
        # Read the YAML file
        with open('config.yaml', 'r') as yaml_file:
            config_data = yaml.safe_load(yaml_file)
        
        # YAML file sections
        Optimization_Problem = config_data["Optimization_Problem"]

        return Optimization_Problem["sim_time"]
        

sim_time = yaml_handling()

###################################################################
# Reading bag files

b = bagreader('simulation.bag')

uav_state = b.message_by_topic('/uav1/estimation_manager/uav_state')
uav_state
hw_cmd = b.message_by_topic('/uav1/mavros/setpoint_raw/attitude')
hw_cmd
target_info = b.message_by_topic('/uav1/target_estimate')
target_info

file = open('simulation/uav1-estimation_manager-uav_state.csv')
type(file)

file2 = open('simulation/uav1-mavros-setpoint_raw-attitude.csv')
type(file2)

file3 = open('simulation/uav1-target_estimate.csv')
type(file3)

###################################################################
# Auxiliary Function
def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

###################################################################
# Processing states data

csvreader = csv.reader(file)

header = []
header = next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)
rows=np.array(rows)

time_vec = rows[:, 0]
time_vec = time_vec.astype(float)
slicer = (sim_time / (time_vec[time_vec.shape[0] - 1] - time_vec[0])) * time_vec.shape[0]
slicer = int(np.round(slicer))
# slicer = -1
time_vec_states = time_vec[0:slicer]

# EULER ANGLES
columns_to_extract = [13, 14, 15, 16]

quaternions_vec = rows[0:slicer, columns_to_extract]
quaternions_vec = quaternions_vec.astype(float)


states = np.zeros((quaternions_vec.shape[0], 9))

for k in range(quaternions_vec.shape[0]):
    states[k, 6:9] = euler_from_quaternion(quaternions_vec[k, 0], quaternions_vec[k, 1],
                                                   quaternions_vec[k, 2], quaternions_vec[k, 3])
    
# VELOCITIES
columns_to_extract = [17, 18, 19]

velocities_vec = rows[0:slicer, columns_to_extract]
velocities_vec = velocities_vec.astype(float)

for k in range(velocities_vec.shape[0]):
    states[k, 3:6] = velocities_vec[k, :]

# POSITIONS

columns_to_extract = [10, 11, 12]

positions_vec = rows[0:slicer, columns_to_extract]
positions_vec = positions_vec.astype(float)

for k in range(positions_vec.shape[0]):
    states[k, 0:3] = positions_vec[k, :]

states = np.transpose(states)

##################################################################
# Processing controls data
csvreader = csv.reader(file2)

header = []
header = next(csvreader)


rows = []
for row in csvreader:
    rows.append(row)
rows=np.array(rows)

time_vec = rows[:, 0]
time_vec = time_vec.astype(float)
slicer = (sim_time / (time_vec[time_vec.shape[0] - 1] - time_vec[0])) * time_vec.shape[0]
slicer = int(np.round(slicer))
time_vec_controls = time_vec[0:slicer]

columns_to_extract = [6, 7, 8, 9]

euler_cmd = rows[0:slicer, columns_to_extract]
euler_cmd = euler_cmd.astype(float)

controls = np.zeros((euler_cmd.shape[0], 4))

for k in range(euler_cmd.shape[0]):
    controls[k, 1:4] = euler_from_quaternion(euler_cmd[k, 0], euler_cmd[k, 1],
                                                   euler_cmd[k, 2], euler_cmd[k, 3])
    controls[k, 0] = rows[k, 13]

controls = np.transpose(controls)

###################################################################
# Arrays Obtained so far
states
controls

# Resize states to match controls

target_shape = (9, (controls.shape[1] + 1))

# states = resize(states, target_shape, mode='reflect', anti_aliasing=True)


###############################################################################
# Target Position

# target_positions = np.zeros((3, int(sim_time/0.2)))
# target_positions[0, 0] = 10
# target_positions[1, 0] = 10
# # target_positions[0, 0] = 10
# # target_positions[1, 0] = 10
# for k in range(int(sim_time/0.2) - 1):
#     target_positions[0, k + 1] = target_positions[0, k] + 0.2
#     target_positions[1, k + 1] = target_positions[1, k] + 0.2

# x_original = np.linspace(0, 1, int(sim_time/0.2))
# x_new = np.linspace(0, 1, states.shape[1])

# target_positions_final = np.zeros((3, states.shape[1]))

# for i in range(target_positions.shape[0]):
#     interpolator = interp.interp1d(x_original, target_positions[i], kind='linear')
#     target_positions_final[i] = interpolator(x_new)


####################################################################################
# Target

csvreader = csv.reader(file3)

header = []
header = next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)
rows=np.array(rows)

# Time Vec
time_vec = rows[:, 0]
time_vec = time_vec.astype(float)
slicer = (sim_time / (time_vec[time_vec.shape[0] - 1] - time_vec[0])) * time_vec.shape[0]
slicer = int(np.round(slicer))
# slicer = -1
time_vec_target = time_vec[0:slicer]

# Target Positions
columns_to_extract = [3, 4]
columns_to_extract_2 = [5, 6]

vx = rows[:, 5].astype(float)
vy = rows[:, 6].astype(float)
# print(vx)
# input()
# print(vy)
# input()
# print(np.sqrt((vx ** 2) + (vy ** 2)))
# input()
# slicer =

target_positions_final = np.zeros((3, slicer))
target_speed = np.zeros((2, slicer))

target_positions_final[0:2, :] = rows[0:slicer, columns_to_extract].transpose()
target_positions_final[0:2, :] = target_positions_final[0:2, :].astype(float)
target_speed[0:2, :] = rows[0:slicer, columns_to_extract_2].transpose()
target_speed[0:2, :] = target_speed[0:slicer, :].astype(float)

#############
x_original = np.linspace(0, 1, target_positions_final.shape[1])
x_new = np.linspace(0, 1, states.shape[1])

target_positions_final_new = np.zeros((3, states.shape[1]))
target_speed_final_new = np.zeros((2, states.shape[1]))

for i in range(target_positions_final.shape[0]):
    interpolator = interp.interp1d(x_original, target_positions_final[i], kind='linear')
    target_positions_final_new[i] = interpolator(x_new)
for i in range(target_speed.shape[0]):
    interpolator = interp.interp1d(x_original, target_speed[i], kind='linear')
    target_speed_final_new[i] = interpolator(x_new)
#############

states
controls
target_positions_final_new
# print(states.shape)
# print(controls.shape)
# print(target_positions_final_new.shape)
# input()

# PLOTS
graph = graphs.plot_graphs_class()
graph.set_variables(target_positions_final_new, states, controls, controls.shape[1], time_vec_states, time_vec_controls, time_vec_target, target_speed_final_new)
graph.plot_starter()


# Recording command:
# rosbag record -O simulation /uav1/estimation_manager/uav_state /uav1/hw_api/attitude_cmd /uav1/target_estimate