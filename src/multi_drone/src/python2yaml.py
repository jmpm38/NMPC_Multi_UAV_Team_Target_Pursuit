import numpy as np
import yaml
from math import pi as pi

# Configuration data
uav_num = 3
config_data = {
    "Optimization_Problem": {
        "N_horizon": 25,
        "dt": 0.2,#0.2,
        "Decentralized": True,
        "sim_time" : 120,
        "fov_terms" : True,
        "formation_terms" : True
    },
    "UAVs_Initial_State": {
        # "x_initials": np.array([[0, 5, 15],
        #                         [0, -5, 1],
        #                         [5, 5, 5],
        #                         [0, 0, 0],
        #                         [0, 0, 0],
        #                         [0, 0, 0],
        #                         [0, 0, 0],
        #                         [0, 0, 0],
        #                         [0, 0, 0]]).tolist(),
        "x_initials": np.array([[0],
                                [0],
                                [5],
                                [0],
                                [0],
                                [0],
                                [0],
                                [0],
                                [0]]).tolist(),
        # "x_initials": np.array([[3.52139002],
        #                         [-4.53143484],
        #                         [6.17198649],
        #                         [0.36520742],
        #                         [0.29265713],
        #                         [1.24068225],
        #                         [-0.25657635],
        #                         [0.09249232],
        #                         [-0.18729148]]).tolist(),
        "uav_num": uav_num
    },

    # "Weights": {
    #     "Q1" : 1000, # standoff distance weight
    #     # "Q2" : np.array([1000000000, 100000000, 1000000000, 0]).tolist(), # Thrust acc, Roll, Pitch, Yaw
    #     "Q2" : np.array([10000000000, 1000000000, 1000000000, 1000000000000]).tolist(), # Thrust acc, Roll, Pitch, Yaw
    #     # "Q2" : np.array([100000000, 1000000000, 1000000000, 0]).tolist(), # Thrust acc, Roll, Pitch, Yaw
    #     "Q3" : 100000000000, # Center of the camera FOV pointing towards the target
    #     "Q4" : 1000, # Fov dot
    #     "Q5" : 50, # Angular separation
    #     "Q6" : 1000000000000, # FOV constraints slack variable
    #     "Q7" : 10000000000, # Height lower bound slack variable
    #     "Q8" : 1000000000 # Following yaw reference
    # },
    "Weights": {
        "Q1" : 0.1, # standoff distance weight
        # "Q2" : np.array([100000, 10000, 10000, 100000000]).tolist(), # Thrust acc, Roll, Pitch, Yaw # 10000000
        "Q2" : np.array([10000, 1000, 1000, 100000]).tolist(), # Thrust acc, Roll, Pitch, Yaw # 10000000
        # "Q2" : np.array([10000, 1000, 1000, 10000000]).tolist(), # Thrust acc, Roll, Pitch, Yaw # 10000000
        "Q3" : 100000000, # Center of the camera FOV pointing towards the target
        "Q4" : 1000, # Fov dot
        "Q5" : 100000, # Fisher Information Matrix
        "Q6" : 100000000, # FOV constraints slack variable
        "Q7" : 1000000, # Height lower bound slack variable
        "Q8" : 1000000 # RPY Rate
    },

    "OCP_Parameters": {
        "Standoff_distance" : 10,
        "U_ref" : np.array([9.81, 0, 0, 0]).tolist(), # T_acc, Pitch, Roll, Yaw
        # "U_ref" : np.array([(2*9.81), 0, 0, 0]).tolist(), # T_acc, Pitch, Roll, Yaw
        "alpha_v" : pi/4,
        "alpha_h" : pi/4,
        "z_rotation_angle" : -1.745329, #2*pi/3,
        "z_axis_camera" : np.array([0, 0, 1]).tolist(),
        "h_min" : 5,
        # "T_max" : 42,#13.5,
        "T_max" : 20.962,#13.5,
        "T_min" : 8.5,
        "pitch_max" : 0.2,#0.5,
        "roll_max": 0.2,#0.5,
        "safety_distance" : 2,
        "desired_separation_angle" : 2*pi/uav_num
    },

    "Target_Parameters": {
        "target_initial_position" : np.array([20, 20, 0]).tolist()
    },

    "Obstacle_Parameters": {
        # "obstacles_positions" : np.array([[5.9, 4.4, 17],
        #                                   [6.8, 12.2, 20],
        #                                   [7.8, 7, 24]]).tolist(),
        "obstacles_positions" : np.array([[16.5, 23],
                                          [19.5, 25],
                                          [5, 3]]).tolist(),
        "obs_radius" : 0.75,
        "safety_radius" : 2
    }
}

# Write the dictionary to a YAML file
with open('config.yaml', 'w') as yaml_file:
    yaml.dump(config_data, yaml_file, default_flow_style=False)
