from bagpy import bagreader
import csv
import numpy as np
import math
from skimage.transform import resize
import target_graphs2 as graphs
from scipy.interpolate import interp1d
import yaml

uav_number = 1
graph = graphs.plot_graphs_class()

def yaml_handling():
        # Read the YAML file
        with open('src/multi_drone/config/config.yaml', 'r') as yaml_file:
            config_data = yaml.safe_load(yaml_file)
        
        # YAML file sections
        Optimization_Problem = config_data["Optimization_Problem"]

        return Optimization_Problem["sim_time"]
        

sim_time = yaml_handling()
graph = graphs.plot_graphs_class()

sim_number = 5
sim_name = str(sim_number)

for uav_number in range(1, 4, 1):
    ###################################################################
    # Reading bag files
    if(uav_number == 1):
        b = bagreader('uav1/simulation_1_' + sim_name + '.bag')

        uav_state = b.message_by_topic('/uav1/estimation_manager/uav_state')
        uav_state
        hw_cmd = b.message_by_topic('/uav1/hw_api/attitude_cmd')
        hw_cmd
        target_info = b.message_by_topic('/uav1/target_estimate')
        target_info
        alpha_slack_info = b.message_by_topic('/uav1/alpha_slack')

        file = open('uav1/simulation_1_' + sim_name + '/uav1-estimation_manager-uav_state.csv')
        type(file)

        file2 = open('uav1/simulation_1_' + sim_name + '/uav1-hw_api-attitude_cmd.csv')
        type(file2)

        file3 = open('uav1/simulation_1_' + sim_name + '/uav1-target_estimate.csv')
        type(file3)

        file4 = open('uav1/simulation_1_' + sim_name + '/uav1-alpha_slack.csv')
        type(file4)

    if (uav_number == 2):
        b = bagreader('uav2/simulation_2_' + sim_name + '.bag')

        uav_state = b.message_by_topic('/uav2/estimation_manager/uav_state')
        uav_state
        hw_cmd = b.message_by_topic('/uav2/hw_api/attitude_cmd')
        hw_cmd
        target_info = b.message_by_topic('/uav1/target_estimate')
        target_info
        alpha_slack_info = b.message_by_topic('/uav2/alpha_slack')

        file = open('uav2/simulation_2_' + sim_name + '/uav2-estimation_manager-uav_state.csv')
        type(file)

        file2 = open('uav2/simulation_2_' + sim_name + '/uav2-hw_api-attitude_cmd.csv')
        type(file2)

        file3 = open('uav2/simulation_2_' + sim_name + '/uav1-target_estimate.csv')
        type(file3)

        file4 = open('uav2/simulation_2_' + sim_name + '/uav2-alpha_slack.csv')
        type(file4)

    if (uav_number == 3):
        b = bagreader('uav3/simulation_3_' + sim_name + '.bag')

        uav_state = b.message_by_topic('/uav3/estimation_manager/uav_state')
        uav_state
        hw_cmd = b.message_by_topic('/uav3/hw_api/attitude_cmd')
        hw_cmd
        target_info = b.message_by_topic('/uav1/target_estimate')
        target_info
        alpha_slack_info = b.message_by_topic('/uav3/alpha_slack')

        file = open('uav3/simulation_3_' + sim_name + '/uav3-estimation_manager-uav_state.csv')
        type(file)

        file2 = open('uav3/simulation_3_' + sim_name + '/uav3-hw_api-attitude_cmd.csv')
        type(file2)

        file3 = open('uav3/simulation_3_' + sim_name + '/uav1-target_estimate.csv')
        type(file3)

        file4 = open('uav3/simulation_3_' + sim_name + '/uav3-alpha_slack.csv')
        type(file4)

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
    #TODO: edit states such that the first line is the time; also check target_speed as well. Finally plot slack variables and plot alpha_desired vs alpha and
    # angular separation between UAVs. Do that by having extra functions in target_graphs2
    csvreader = csv.reader(file)

    header = []
    header = next(csvreader)

    rows = []
    for row in csvreader:
        rows.append(row)
    rows=np.array(rows)

    time_vec = rows[:, 0]
    time_vec = time_vec.astype(float)

    time_vec_states = time_vec[:]

    # EULER ANGLES
    columns_to_extract = [13, 14, 15, 16]

    quaternions_vec = rows[:, columns_to_extract]
    quaternions_vec = quaternions_vec.astype(float)


    states = np.zeros((quaternions_vec.shape[0], 11))

    for k in range(quaternions_vec.shape[0]):
        states[k, 7:10] = euler_from_quaternion(quaternions_vec[k, 0], quaternions_vec[k, 1],
                                                    quaternions_vec[k, 2], quaternions_vec[k, 3])
        
    # VELOCITIES
    columns_to_extract = [0, 17, 18, 19]

    velocities_vec = rows[:, columns_to_extract]
    velocities_vec = velocities_vec.astype(float)

    for k in range(velocities_vec.shape[0]):
        states[k, 0] = velocities_vec[k, 0]
        states[k, 4:7] = velocities_vec[k, 1:4]

    # POSITIONS

    columns_to_extract = [10, 11, 12]

    positions_vec = rows[:, columns_to_extract]
    positions_vec = positions_vec.astype(float)

    for k in range(positions_vec.shape[0]):
        states[k, 1:4] = positions_vec[k, :]

    states[:, 10] = np.cos(states[:, 9])
    states[:, 9] = np.sin(states[:, 9])

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
    time_vec_controls = time_vec[:]

    columns_to_extract = [3, 4, 5, 6]

    euler_cmd = rows[:, columns_to_extract]
    euler_cmd = euler_cmd.astype(float)

    controls = np.zeros((euler_cmd.shape[0], 5))

    for k in range(euler_cmd.shape[0]):
        controls[k, 0] = rows[k, 0]
        controls[k, 2:5] = euler_from_quaternion(euler_cmd[k, 0], euler_cmd[k, 1],
                                                    euler_cmd[k, 2], euler_cmd[k, 3])
        controls[k, 1] = rows[k, 7]

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
    # slicer = -1
    time_vec_target = time_vec[:]

    # Target Positions
    columns_to_extract = [0, 3, 4]
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

    target_positions_final = np.zeros((6, time_vec.shape[0]))

    target_positions_final[0:3, :] = rows[:, columns_to_extract].transpose()
    target_positions_final[0:3, :] = target_positions_final[0:3, :].astype(float)
    target_positions_final[4:6, :] = rows[:, columns_to_extract_2].transpose()
    target_positions_final[4:6, :] = target_positions_final[4:6, :].astype(float)

    #############
    # x_original = np.linspace(0, 1, target_positions_final.shape[1])
    # x_new = np.linspace(0, 1, states.shape[1])

    # target_positions_final_new = np.zeros((3, states.shape[1]))

    # for i in range(target_positions_final.shape[0]):
    #     interpolator = interp.interp1d(x_original, target_positions_final[i], kind='linear')
    #     target_positions_final_new[i] = interpolator(x_new)
    #############

    ###########################################################################3
    # ALPHA SLACK
    ##################################################################
    # Processing controls data
    csvreader = csv.reader(file4)

    header = []
    header = next(csvreader)

    rows = []
    for row in csvreader:
        rows.append(row)
    rows=np.array(rows)


    time_vec = rows[:, 0]
    time_vec = time_vec.astype(float)

    columns_to_extract = [0, 3, 4, 5] # S1, S2, S3
    column_alpha = [0, 6] # Alpha_desired

    slack_variables = rows[0:, columns_to_extract]
    slack_variables = slack_variables.astype(float)

    alpha_desired = rows[0:, column_alpha]
    alpha_desired = alpha_desired.astype(float)

    slack_variables = np.transpose(slack_variables)
    alpha_desired = np.transpose(alpha_desired)

    columnsss = [0, 7]
    solving_time =  rows[:, columnsss]
    solving_time = solving_time.astype(float)
    solving_time = np.transpose(solving_time)

    #########################################################################3

    # print(states.shape)
    # print(controls.shape)
    # print(target_positions_final.shape)#_new
    # print(slack_variables.shape)
    # print(alpha_desired.shape)

    ###########################################################################
    arrays = [states, controls, target_positions_final, slack_variables, alpha_desired]

    # Step 1: Extract time rows
    time_rows = [arr[0, :] for arr in arrays]

    # Step 2: Determine common time axis
    start_time = max([min(time) for time in time_rows])
    # print("start_time:")
    # print(start_time)
    end_time = min(max(time) for time in time_rows)#min([min(max(time) for time in time_rows), sim_time])
    # print("end_time:")
    # print(end_time)

    # Define the resolution of the common time axis (number of points)
    # You may want to set this to match the highest resolution in your data
    min_num_columns = max(arr.shape[1] for arr in arrays)
    common_time = np.linspace(start_time, end_time, num=min_num_columns)

    # Step 3: Interpolate data to match common time axis
    interpolated_arrays = []

    for arr, time in zip(arrays, time_rows):
        interp_func = interp1d(time, arr, axis=1, bounds_error=False, fill_value='extrapolate')
        interpolated_arr = interp_func(common_time)
        interpolated_arrays.append(interpolated_arr)

    # for i, interpolated_arr in enumerate(interpolated_arrays):
    #     print(f"Array {i+1} shape after interpolation: {interpolated_arr.shape}")


    ###########################################################################
    interpolated_arrays[0][9, :] = np.arctan2(interpolated_arrays[0][9, :], interpolated_arrays[0][10, :])


    # PLOTS
    print("UAV: " + str(uav_number - 1))
    graph.set_variables(interpolated_arrays[2][1:4, :], interpolated_arrays[0][1:10, :], interpolated_arrays[1][1:, :], controls.shape[1], interpolated_arrays[0][0, :] - interpolated_arrays[0][0, 0], interpolated_arrays[3][1:, :], interpolated_arrays[4][1:, :], uav_number - 1, interpolated_arrays[2][4:6, :])
    graph.plot_solving_time(solving_time)
    graph.plot_starter()


    # Recording command:
    # rosbag record -O simulation /uav1/estimation_manager/uav_state /uav1/hw_api/attitude_cmd /uav1/target_estimate
