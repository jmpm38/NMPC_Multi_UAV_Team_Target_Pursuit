import casadi as ca
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from casadi import sin, cos, pi, tan
import math
import yaml

# Right now the class needs to be initialized -> set_variables (receive target positions, UAV states, UAV control inputs) -> plot_starter (calls plot3D after input())

class plot_graphs_class:
    
    def __init__(self):
        self.yaml_handling()

    def plot_starter(self):
        self.fun_plot_uav_target_XY()
        self.fun_plot_uav_xyz()
        self.fun_plot_uav_target_dist()
        self.fun_plot_uav_roll_roll_ref()
        self.fun_plot_uav_pitch_pitch_ref()
        self.fun_plot_uav_yaw_yaw_ref()
        # self.fun_plot_uav_speed()
        # if(self.target_speed != 0):
        # self.fun_plot_uav_vxvyvz()
        self.fun_plot_obstacle_safety()
        self.fun_plot_occlusion()

        self.fun_plot_fov()
        self.fun_plot_input_constr()

        if(self.uav_num>1):
            self.fun_plot_distance_between_uavs()
            self.fun_plot_angular_separations_uavs()

        # show plots
        self.plot_show()

    def yaml_handling(self):
            # Read the YAML file
            with open('config.yaml', 'r') as yaml_file:
                config_data = yaml.safe_load(yaml_file)
            
            # YAML file sections
            Optimization_Problem = config_data["Optimization_Problem"]
            UAVs_Initial_State = config_data["UAVs_Initial_State"]
            Weights = config_data["Weights"]
            OCP_Parameters = config_data["OCP_Parameters"]
            Obstacle_Parameters = config_data["Obstacle_Parameters"]

            self.N_horizon = Optimization_Problem["N_horizon"]
            self.dt = Optimization_Problem["dt"]
            self.decentralized = Optimization_Problem["Decentralized"]
            self.sim_time = Optimization_Problem["sim_time"]
            self.time_steps = int(self.sim_time/self.dt)
            self.fov_terms = Optimization_Problem["fov_terms"]
            self.formation_terms = Optimization_Problem["formation_terms"]

            self.uav_num = 1#UAVs_Initial_State["uav_num"]

            self.Q1 = Weights["Q1"]
            self.Q2 = np.array(Weights['Q2'])
            self.Q3 = Weights["Q3"]
            self.Q4 = Weights["Q4"]
            self.Q5 = Weights["Q5"]
            self.Q6 = Weights["Q6"]
            
            self.Standoff_distance = OCP_Parameters['Standoff_distance']
            self.U_ref = np.array(OCP_Parameters['U_ref'])
            self.alpha_v = OCP_Parameters['alpha_v']
            self.alpha_h = OCP_Parameters['alpha_h']
            self.z_rotation_angle = OCP_Parameters['z_rotation_angle']
            self.z_axis_camera = np.array(OCP_Parameters['z_axis_camera'])
            self.h_min = OCP_Parameters['h_min']
            self.T_max = OCP_Parameters['T_max']
            self.T_min = OCP_Parameters['T_min']
            self.pitch_max = OCP_Parameters['pitch_max']
            self.roll_max = OCP_Parameters['roll_max']
            self.safety_distance = OCP_Parameters['safety_distance']
            self.desired_separation_angle = OCP_Parameters['desired_separation_angle']

            self.obstacles_positions = np.array(Obstacle_Parameters['obstacles_positions'])
            self.obs_radius = Obstacle_Parameters['obs_radius']
            self.safety_radius = Obstacle_Parameters['safety_radius']

    def set_variables(self, target_positions, X_states, U_controls, horizon, time_vec_states, time_vec_controls, time_vec_target, target_speed=0):
        self.N_horizon = horizon

        self.target_position = target_positions # 3, self.time_steps + 1
        self.target_speed = target_speed
        self.X_states = X_states # 9*self.uav_num , self.time_steps + 1
        self.U_controls = U_controls # 4*self.uav_num , self.time_steps
        self.time_vec_state = time_vec_states
        self.time_vec_control = time_vec_controls
        self.time_steps = self.X_states.shape[1]
        self.time_vec_target = time_vec_target

    def fun_plot_uav_target_XY(self):
        plt.figure('xy_target')
    
        # Plot UAV positions
        i = 1
        for uav_i in range(0, self.uav_num * 9, 9):
            u = np.cos(self.X_states[8 + uav_i, :])
            v = np.sin(self.X_states[8 + uav_i, :])

            plt.scatter(self.X_states[0 + uav_i, :], self.X_states[1 + uav_i, :], label=('UAV ' + str(i) + ' XY position'))
            # plt.quiver(self.X_states[0 + uav_i, :], self.X_states[1 + uav_i, :], u, v, angles='xy', scale_units='xy', scale=0.5, color='red')
            i += 1
        
        # Plot target position
        plt.scatter(self.target_position[0, :], self.target_position[1, :], color='r', label="target_position", marker="*")
        
        # Add a circle (example)
        # Center of the circle is at the target position, radius is 10 (change as needed)
        circle = np.empty((self.obstacles_positions.shape[1], 1))
        for k_circ in range(circle.shape[0]):
            circle = patches.Circle((self.obstacles_positions[0 ,k_circ], self.obstacles_positions[1 ,k_circ]), radius=self.obs_radius, edgecolor='blue', facecolor='none', linestyle='--', linewidth=1)
            circle2 = patches.Circle((self.obstacles_positions[0 ,k_circ], self.obstacles_positions[1 ,k_circ]), radius=self.safety_radius, edgecolor='red', facecolor='none', linestyle='--', linewidth=1)
            
            # Get current axes and add the circle
            plt.gca().add_patch(circle)
            plt.gca().add_patch(circle2)

            # Make sure the aspect ratio is equal so the circle isn't distorted
            plt.gca().set_aspect('equal', adjustable='box')

        # # Add labels, title, and legend
        # plt.gca().invert_yaxis()
        # # Get the current axis
        # ax = plt.gca()

        # # Move x-axis to the top
        # ax.xaxis.set_ticks_position('top')
        # ax.xaxis.set_label_position('top')

        plt.legend()
        plt.ylabel('y (m)')
        plt.xlabel('x (m)')
        plt.title("UAV position and Target position on the XY plane")

    def fun_plot_uav_xyz(self):
        i = 1
        #plt.figure("z")
        for uav_i in range(0, self.uav_num*9, 9):
            plt.figure('uav ' + str(i) + ' xyz')
            plt.plot(self.time_vec_state, self.X_states[0 + uav_i, :], label=('X' + str(i)))
            plt.plot(self.time_vec_state, self.X_states[1 + uav_i, :], label=('Y' + str(i)))
            plt.plot(self.time_vec_state, self.X_states[2 + uav_i, :], label=('Z' + str(i)))
            # plt.plot(self.time_vec_state, self.target_position[0, :], label=('X_t' + str(i)))
            # plt.plot(self.time_vec_state, self.target_position[1, :], label=('Y_t' + str(i)))
            # plt.plot(self.time_vec_state, self.target_position[2, :], label=('Z_t' + str(i)))
            plt.axhline(y=self.h_min, color='r', linestyle='--')
            # plt.axhline(y=30, color='r', linestyle='--')
            plt.legend()
            plt.ylabel('x, y, z (m)')
            plt.xlabel('Time (s)')
            plt.title("UAV " + str(i) + " x, y, z coordinates")
            i = i + 1

    def fun_plot_uav_target_dist(self):
        plt.figure("distance to target")
        i = 1
        plt.axhline(y=self.Standoff_distance, color='b', linestyle=':', label="Desired distance to Target") 
        for uav_i in range(0, self.uav_num*9, 9):
            dist_vec = []
            for k in range(0,self.time_steps):
                current_uav_pos = [self.X_states[0+uav_i, k], self.X_states[1+uav_i, k], self.X_states[2+uav_i, k]]
                current_target_pos = [self.target_position[0, k], self.target_position[1, k], self.target_position[2, k]]
                dist = math.dist(current_uav_pos, current_target_pos)
                dist_vec.append(dist)
            plt.plot(self.time_vec_state, dist_vec, label=("UAV " + str(i) + " distance to the target"))
            i = i + 1
        plt.legend()
        plt.ylabel('Distance (m)')
        plt.xlabel('Time (s)')
        plt.title("Distance between UAVs and the target")

    def fun_plot_input_constr(self):
        i = 1
        print("a")
        for uav_i in range(0, self.uav_num*4, 4):
            fig, axs = plt.subplots(4)
            fig.suptitle('Control Inputs UAV ' + str(i))
            axs[0].plot(self.time_vec_control, self.U_controls[0+uav_i, :])
            # axs[0].axhline(y=self.T_min, color='r', linestyle=':', label="Thrust boundaries")
            # axs[0].axhline(y=self.T_max, color='r', linestyle=':')
            # axs[0].axhline(y=self.U_ref[0], color='b', linestyle=':', label="Thrust reference")
            axs[0].axhline(y=0, color='r', linestyle=':', label="Thrust boundaries")
            axs[0].axhline(y=1, color='r', linestyle=':')
            axs[0].axhline(y=0.5433419696, color='b', linestyle=':', label="Hover Throttle")
            axs[0].legend()
            axs[0].set_title('Throttle')
            axs[0].set(ylabel='Throttle')
            axs[1].plot(self.time_vec_control, self.U_controls[1+uav_i, :])
            axs[1].axhline(y=-self.pitch_max, color='r', linestyle=':', label="Boundaries")
            axs[1].axhline(y=self.pitch_max, color='r', linestyle=':')
            axs[1].axhline(y=self.U_ref[1], color='b', linestyle=':', label="Roll reference")
            axs[1].legend()
            axs[1].set_title('Roll Angle')
            axs[1].set(ylabel='Angle (rad)')
            axs[2].plot(self.time_vec_control, self.U_controls[2+uav_i, :])
            axs[2].axhline(y=-self.roll_max, color='r', linestyle=':', label="Boundaries")
            axs[2].axhline(y=self.roll_max, color='r', linestyle=':')
            axs[2].axhline(y=self.U_ref[2], color='b', linestyle=':', label="Pitch reference")
            axs[2].legend()
            axs[2].set_title('Pitch Angle')
            axs[2].set(ylabel='Angle (rad)')
            axs[3].plot(self.time_vec_control, self.U_controls[3+uav_i, :])
            axs[3].axhline(y=-pi, color='r', linestyle='-')
            axs[3].axhline(y=pi, color='r', linestyle='-')
            axs[3].set_title('Yaw Angle')
            axs[3].set(ylabel='Angle (rad)', xlabel='Time (s)')
            i = i + 1

    def fun_plot_fov(self):
        i = 1
        print("a")
        for uav_i in range(0, self.uav_num*4, 4):
            fig, axs = plt.subplots(4)
            fig.suptitle('Control Inputs UAV ' + str(i))
            axs[0].plot(self.time_vec_control, self.U_controls[0+uav_i, :])
            # axs[0].axhline(y=self.T_min, color='r', linestyle=':', label="Thrust boundaries")
            # axs[0].axhline(y=self.T_max, color='r', linestyle=':')
            # axs[0].axhline(y=self.U_ref[0], color='b', linestyle=':', label="Thrust reference")
            axs[0].axhline(y=0, color='r', linestyle=':', label="Thrust boundaries")
            axs[0].axhline(y=1, color='r', linestyle=':')
            axs[0].axhline(y=0.5433419696, color='b', linestyle=':', label="Hover Throttle")
            axs[0].legend()
            axs[0].set_title('Throttle')
            axs[0].set(ylabel='Throttle')
            axs[1].plot(self.time_vec_control, self.U_controls[1+uav_i, :])
            axs[1].axhline(y=-self.pitch_max, color='r', linestyle=':', label="Boundaries")
            axs[1].axhline(y=self.pitch_max, color='r', linestyle=':')
            axs[1].axhline(y=self.U_ref[1], color='b', linestyle=':', label="Roll reference")
            axs[1].legend()
            axs[1].set_title('Roll Angle')
            axs[1].set(ylabel='Angle (rad)')
            axs[2].plot(self.time_vec_control, self.U_controls[2+uav_i, :])
            axs[2].axhline(y=-self.roll_max, color='r', linestyle=':', label="Boundaries")
            axs[2].axhline(y=self.roll_max, color='r', linestyle=':')
            axs[2].axhline(y=self.U_ref[2], color='b', linestyle=':', label="Pitch reference")
            axs[2].legend()
            axs[2].set_title('Pitch Angle')
            axs[2].set(ylabel='Angle (rad)')
            axs[3].plot(self.time_vec_control, self.U_controls[3+uav_i, :])
            axs[3].axhline(y=-pi, color='r', linestyle='-')
            axs[3].axhline(y=pi, color='r', linestyle='-')
            axs[3].set_title('Yaw Angle')
            axs[3].set(ylabel='Angle (rad)', xlabel='Time (s)')
            i = i + 1

    def fun_plot_fov(self):
        rotation_matrix_b2w = np.zeros((3, 3, self.time_steps))
        camera_rotation_matrix_b2c = np.zeros((3, 3, self.time_steps))
        rotation_matrix_w2b = np.zeros((3, 3, self.time_steps))            

        plot_arrays = np.zeros((3, self.time_steps))
        iteration = 1

        for uav_i in range(0, self.uav_num * 9, 9):
            # Set up b2w and b2c rotation matrices throughout the horizon
            for k in range(0,self.time_steps):

                rotation_matrix_b2w[0, 0, k] = cos(self.X_states[8 + uav_i, k]) * cos(self.X_states[7 + uav_i, k])
                rotation_matrix_b2w[0, 1, k] = cos(self.X_states[8 + uav_i, k]) * sin(self.X_states[7 + uav_i, k]) * sin(self.X_states[6 + uav_i, k]) - sin(self.X_states[8 + uav_i, k]) * cos(self.X_states[6 + uav_i, k])
                rotation_matrix_b2w[0, 2, k] = cos(self.X_states[8 + uav_i, k]) * sin(self.X_states[7 + uav_i, k]) * cos(self.X_states[6 + uav_i, k]) + sin(self.X_states[8 + uav_i, k]) * sin(self.X_states[6 + uav_i, k])

                rotation_matrix_b2w[1, 0, k] = sin(self.X_states[8 + uav_i, k]) * cos(self.X_states[7 + uav_i, k])
                rotation_matrix_b2w[1, 1, k] = sin(self.X_states[8 + uav_i, k]) * sin(self.X_states[7 + uav_i, k]) * sin(self.X_states[6 + uav_i, k]) + cos(self.X_states[8 + uav_i, k]) * cos(self.X_states[6 + uav_i, k])
                rotation_matrix_b2w[1, 2, k] = sin(self.X_states[8 + uav_i, k]) * sin(self.X_states[7 + uav_i, k]) * cos(self.X_states[6 + uav_i, k]) - cos(self.X_states[8 + uav_i, k]) * sin(self.X_states[6 + uav_i, k])

                rotation_matrix_b2w[2, 0, k] = -sin(self.X_states[7 + uav_i, k])
                rotation_matrix_b2w[2, 1, k] = cos(self.X_states[7 + uav_i, k]) * sin(self.X_states[6 + uav_i, k])
                rotation_matrix_b2w[2, 2, k] = cos(self.X_states[7 + uav_i, k]) * cos(self.X_states[6 + uav_i, k])

                camera_rotation_matrix_b2c[0, 0, k] = cos(self.z_rotation_angle)
                camera_rotation_matrix_b2c[0, 1, k] = 0
                camera_rotation_matrix_b2c[0, 2, k] = sin(self.z_rotation_angle)

                camera_rotation_matrix_b2c[1, 0, k] = 0
                camera_rotation_matrix_b2c[1, 1, k] = 1
                camera_rotation_matrix_b2c[1, 2, k] = 0

                camera_rotation_matrix_b2c[2, 0, k] = -sin(self.z_rotation_angle)
                camera_rotation_matrix_b2c[2, 1, k] = 0
                camera_rotation_matrix_b2c[2, 2, k] = cos(self.z_rotation_angle)

            # Set up w2b rotation matrix throughout the horizon
            for k in range(0, self.time_steps):
                rotation_matrix_w2b[:, :, k] = np.transpose(rotation_matrix_b2w[:, :, k])

            # Set up b2w and b2c rotation matrices throughout the horizon
            for k in range(0,self.time_steps):
                relative_target_position = np.subtract(self.target_position[:, k], self.X_states[0+uav_i : 3+uav_i, k])
                relative_target_position_BFrame = np.matmul(rotation_matrix_w2b[:, :, k], relative_target_position)
                cam_pos = [0.226, 0, -0.089]
                relative_target_position_BFrame = np.subtract(relative_target_position_BFrame, cam_pos)
                relative_target_position_CFrame = np.matmul(camera_rotation_matrix_b2c[:, :, k], relative_target_position_BFrame)

                relative_target_position_CFrame_norm = np.linalg.norm(relative_target_position_CFrame)
                beta_versor = (1/relative_target_position_CFrame_norm) * relative_target_position_CFrame

                z_camera = [0, 0, 1]
                beta_angle = np.inner(beta_versor, z_camera)
                plot_arrays[0, k] = np.degrees(np.arccos(beta_angle))

                horizontal = [beta_versor[0], 0, beta_versor[2]]
                vertical = [0, beta_versor[1], beta_versor[2]]
                horz_angle = np.inner(horizontal, z_camera)
                vert_angle = np.inner(vertical, z_camera)
                plot_arrays[1, k] = np.degrees(np.arccos(horz_angle))
                plot_arrays[2, k] = np.degrees(np.arccos(vert_angle))

                # horz_limit = cos(self.alpha_h/2)
                # vert_limit = cos(self.alpha_v/2)
                horz_limit = np.degrees(self.alpha_h/2)
                vert_limit = np.degrees(self.alpha_v/2)
                
            fig, axs = plt.subplots(3)
            fig.suptitle('Target Tracking UAV ' + str(iteration))

            axs[0].plot(self.time_vec_state, plot_arrays[0, :], label='Angle')
            axs[0].axhline(y=0, color='b', linestyle=':', label='Desired value (fov center)')
            axs[0].legend()
            axs[0].set(ylabel='Angle(deg)')
            axs[0].set_title("Angle between Target and Camera Center")

            axs[1].plot(self.time_vec_state, plot_arrays[1, :], label='Horizontal Angle')
            axs[1].axhline(y=horz_limit, color='r', linestyle=':', label='FOV Limit')
            axs[1].legend()
            axs[1].set(ylabel='Angle(deg)')

            axs[2].plot(self.time_vec_state, plot_arrays[2, :], label='Vertical Angle')
            axs[2].axhline(y=vert_limit, color='r', linestyle=':', label='FOV Limit')
            axs[2].legend()
            axs[2].set(ylabel='Angle(deg)', xlabel='Time (s)')

            iteration = iteration + 1
        # rotation_matrix_b2w = np.zeros((3, 3, self.time_steps))
        # camera_rotation_matrix_b2c = np.zeros((3, 3, self.time_steps))
        # rotation_matrix_w2b = np.zeros((3, 3, self.time_steps))            

        # plot_arrays = np.zeros((3, self.time_steps))
        # iteration = 1
        # for uav_i in range(0, self.uav_num * 9, 9):
        #     # Set up b2w and b2c rotation matrices throughout the horizon
        #     for k in range(0,self.time_steps):

        #         rotation_matrix_b2w[0, 0, k] = cos(self.X_states[8 + uav_i, k]) * cos(self.X_states[7 + uav_i, k])
        #         rotation_matrix_b2w[0, 1, k] = cos(self.X_states[8 + uav_i, k]) * sin(self.X_states[7 + uav_i, k]) * sin(self.X_states[6 + uav_i, k]) - sin(self.X_states[8 + uav_i, k]) * cos(self.X_states[6 + uav_i, k])
        #         rotation_matrix_b2w[0, 2, k] = cos(self.X_states[8 + uav_i, k]) * sin(self.X_states[7 + uav_i, k]) * cos(self.X_states[6 + uav_i, k]) + sin(self.X_states[8 + uav_i, k]) * sin(self.X_states[6 + uav_i, k])

        #         rotation_matrix_b2w[1, 0, k] = sin(self.X_states[8 + uav_i, k]) * cos(self.X_states[7 + uav_i, k])
        #         rotation_matrix_b2w[1, 1, k] = sin(self.X_states[8 + uav_i, k]) * sin(self.X_states[7 + uav_i, k]) * sin(self.X_states[6 + uav_i, k]) + cos(self.X_states[8 + uav_i, k]) * cos(self.X_states[6 + uav_i, k])
        #         rotation_matrix_b2w[1, 2, k] = sin(self.X_states[8 + uav_i, k]) * sin(self.X_states[7 + uav_i, k]) * cos(self.X_states[6 + uav_i, k]) - cos(self.X_states[8 + uav_i, k]) * sin(self.X_states[6 + uav_i, k])

        #         rotation_matrix_b2w[2, 0, k] = -sin(self.X_states[7 + uav_i, k])
        #         rotation_matrix_b2w[2, 1, k] = cos(self.X_states[7 + uav_i, k]) * sin(self.X_states[6 + uav_i, k])
        #         rotation_matrix_b2w[2, 2, k] = cos(self.X_states[7 + uav_i, k]) * cos(self.X_states[6 + uav_i, k])


        #         camera_rotation_matrix_b2c[0, 0, k] = cos(self.z_rotation_angle)
        #         camera_rotation_matrix_b2c[0, 1, k] = 0
        #         camera_rotation_matrix_b2c[0, 2, k] = sin(self.z_rotation_angle)

        #         camera_rotation_matrix_b2c[1, 0, k] = 0
        #         camera_rotation_matrix_b2c[1, 1, k] = 1
        #         camera_rotation_matrix_b2c[1, 2, k] = 0

        #         camera_rotation_matrix_b2c[2, 0, k] = -sin(self.z_rotation_angle)
        #         camera_rotation_matrix_b2c[2, 1, k] = 0
        #         camera_rotation_matrix_b2c[2, 2, k] = cos(self.z_rotation_angle)

        #     # Set up w2b rotation matrix throughout the horizon
        #     for k in range(0, self.time_steps):
        #         rotation_matrix_w2b[:, :, k] = np.transpose(rotation_matrix_b2w[:, :, k])

        #     # Set up b2w and b2c rotation matrices throughout the horizon
        #     for k in range(0,self.time_steps):
        #         relative_target_position = np.subtract(self.target_position[:, k], self.X_states[0+uav_i : 3+uav_i, k])
        #         relative_target_position_BFrame = np.matmul(rotation_matrix_w2b[:, :, k], relative_target_position)
        #         relative_target_position_CFrame = np.matmul(camera_rotation_matrix_b2c[:, :, k], relative_target_position_BFrame)

        #         relative_target_position_CFrame_norm = np.linalg.norm(relative_target_position_CFrame)
        #         beta_versor = (1/relative_target_position_CFrame_norm) * relative_target_position_CFrame

        #         z_camera = [0, 0, 1]
        #         beta_angle = np.inner(beta_versor, z_camera)
        #         plot_arrays[0, k] = beta_angle

        #         horizontal = [beta_versor[0], 0, beta_versor[2]]
        #         vertical = [0, beta_versor[1], beta_versor[2]]
        #         horz_angle = np.inner(horizontal, z_camera)
        #         vert_angle = np.inner(vertical, z_camera)
        #         plot_arrays[1, k] = horz_angle
        #         plot_arrays[2, k] = vert_angle

        #         horz_limit = cos(self.alpha_h/2)
        #         vert_limit = cos(self.alpha_v/2)
                
        #     fig, axs = plt.subplots(3)
        #     fig.suptitle('Target Tracking UAV ' + str(iteration))

        #     axs[0].plot(self.time_vec_state, plot_arrays[0, :], label='cos(Beta)')
        #     axs[0].axhline(y=1, color='b', linestyle=':', label='Desired value (fov center)')
        #     axs[0].legend()
        #     axs[0].set(ylabel='cos(Beta)')
        #     axs[0].set_title("Cosine of the Angle between target position and camera z axis")

        #     axs[1].plot(self.time_vec_state, plot_arrays[1, :], label='cos(Beta_h)')
        #     axs[1].axhline(y=horz_limit, color='r', linestyle=':', label='boundary')
        #     axs[1].legend()
        #     axs[1].set(ylabel='cos(Beta_h)')

        #     axs[2].plot(self.time_vec_state, plot_arrays[2, :], label='cos(Beta_v)')
        #     axs[2].axhline(y=vert_limit, color='r', linestyle=':', label='boundary')
        #     axs[2].legend()
        #     axs[2].set(ylabel='cos(Beta_v)', xlabel='Time (s)')

    #TODO:
    def fun_plot_distance_between_uavs(self):
        plt.figure("Distance between UAVs")
        i = 1
        plt.axhline(y=self.safety_distance, color='b', linestyle=':', label="Minimum distance between UAVs") 
        combinations = math.comb(self.uav_num, 2)
        distances_vector = np.zeros((combinations, self.time_steps+1))
        aux = 9
        aux1 = 0
        i = 1
        j = 2
        for iter1 in range(0, 9*self.uav_num, 9):
            for iter2 in range(aux, 9*self.uav_num, 9):
                for k in range(0,self.time_steps+1):
                    vector1 = [self.X_states[0+iter1, k], self.X_states[1+iter1, k], self.X_states[2+iter1, k]]
                    vector2 = [self.X_states[0+iter2, k], self.X_states[1+iter2, k], self.X_states[2+iter2, k]]
                    distances_vector[aux1, k] = np.linalg.norm(np.subtract(vector1, vector2))
                plt.plot(self.time_vec_state, distances_vector[aux1, :], label=("Dist UAV " + str(i) + " and UAV " + str(j)))
                aux1 = aux1 + 1
                j = j + 1
            aux = aux + 9
            i = i +1
            j = i + 1
        plt.legend()
        plt.ylabel('Distance (m)')
        plt.xlabel('Time (s)')
        plt.title("Distance between UAVs")
    #TODO:
    def fun_plot_angular_separations_uavs(self):
        plt.figure("Angular distance between UAVs")
        i = 1
        plt.axhline(y=(2*pi/self.uav_num), color='b', linestyle=':', label="Minimum distance") 
        plt.axhline(y=pi, color='r', linestyle=':', label="PI") 
        plt.axhline(y=(pi/2), color='r', linestyle=':', label="PI/2") 
        combinations = math.comb(self.uav_num, 2)
        angle_vector = np.zeros((combinations, self.time_steps+1))
        aux = 9
        aux1 = 0
        i = 1
        j = 2
        for iter1 in range(0, 9*self.uav_num, 9):
            for iter2 in range(aux, 9*self.uav_num, 9):
                for k in range(0,self.time_steps+1):
                    vector1 = np.array([self.X_states[0+iter1, k] - self.target_position[0, k], self.X_states[1+iter1, k] - self.target_position[1, k]])
                    vector2 = np.array([self.X_states[0+iter2, k] - self.target_position[0, k], self.X_states[1+iter2, k] - self.target_position[1, k]])
                    vector1 = vector1 * (1/np.linalg.norm(vector1))
                    vector2 = vector2 * (1/np.linalg.norm(vector2))
                    angle_vector[aux1, k] = np.arccos(np.dot(vector1, vector2))
                plt.plot(self.time_vec_state, angle_vector[aux1, :], label=("Ang dist UAV " + str(i) + " and UAV " + str(j)))
                aux1 = aux1 + 1
                j = j + 1
            aux = aux + 9
            i = i +1
            j = i + 1
        plt.legend()
        plt.ylabel('Angle (rad)')
        plt.xlabel('Time (s)')
        plt.title("Angular distance between UAVs relative to the target")

    def fun_plot_uav_roll_roll_ref(self):
        plt.figure("roll")
        plt.plot(self.time_vec_state, np.degrees(self.X_states[6, :]), label='Roll')
        plt.plot(self.time_vec_control, np.degrees(self.U_controls[1, :]), label='Roll Ref')
        plt.axhline(y=np.degrees(self.roll_max), color='r', linestyle='--')
        plt.axhline(y=-np.degrees(self.roll_max), color='r', linestyle='--')
        plt.axhline(y=self.U_ref[1], color='b', linestyle=':')
        plt.legend()
        plt.ylabel('Angle(deg)')
        plt.xlabel('Time (s)')
        plt.title("Roll/Roll Ref Comparison")

    def fun_plot_uav_pitch_pitch_ref(self):
        plt.figure("pitch")
        plt.plot(self.time_vec_state, np.degrees(self.X_states[7, :]), label='Pitch')
        plt.plot(self.time_vec_control, np.degrees(self.U_controls[2, :]), label='Pitch Ref')
        plt.axhline(y=np.degrees(self.pitch_max), color='r', linestyle='--')
        plt.axhline(y=-np.degrees(self.pitch_max), color='r', linestyle='--')
        plt.axhline(y=self.U_ref[2], color='b', linestyle=':')
        plt.legend()
        plt.ylabel('Angle(deg)')
        plt.xlabel('Time (s)')
        plt.title("Pitch/Pitch Ref Comparison")

    def fun_plot_uav_yaw_yaw_ref(self):
        plt.figure("yaw")
        plt.plot(self.time_vec_state, np.degrees(self.X_states[8, :]), label='Yaw')
        plt.plot(self.time_vec_control, np.degrees(self.U_controls[3, :]), label='Yaw Ref Control')
        reference_yaw = np.zeros((self.time_steps, 1))
        # rotation_matrix_b2w = np.zeros((3, 3, self.time_steps))
        # rotation_matrix_w2b = np.zeros((3, 3, self.time_steps))   
        for k in range(0,self.time_steps):
            # rotation_matrix_b2w[0, 0, k] = cos(self.X_states[8, k]) * cos(self.X_states[7, k])
            # rotation_matrix_b2w[0, 1, k] = cos(self.X_states[8 , k]) * sin(self.X_states[7, k]) * sin(self.X_states[6, k]) - sin(self.X_states[8, k]) * cos(self.X_states[6, k])
            # rotation_matrix_b2w[0, 2, k] = cos(self.X_states[8 , k]) * sin(self.X_states[7, k]) * cos(self.X_states[6, k]) + sin(self.X_states[8, k]) * sin(self.X_states[6, k])

            # rotation_matrix_b2w[1, 0, k] = sin(self.X_states[8, k]) * cos(self.X_states[7, k])
            # rotation_matrix_b2w[1, 1, k] = sin(self.X_states[8, k]) * sin(self.X_states[7, k]) * sin(self.X_states[6, k]) + cos(self.X_states[8, k]) * cos(self.X_states[6, k])
            # rotation_matrix_b2w[1, 2, k] = sin(self.X_states[8, k]) * sin(self.X_states[7, k]) * cos(self.X_states[6, k]) - cos(self.X_states[8, k]) * sin(self.X_states[6, k])

            # rotation_matrix_b2w[2, 0, k] = -sin(self.X_states[7, k])
            # rotation_matrix_b2w[2, 1, k] = cos(self.X_states[7, k]) * sin(self.X_states[6, k])
            # rotation_matrix_b2w[2, 2, k] = cos(self.X_states[7, k]) * cos(self.X_states[6, k])

            # rotation_matrix_w2b[:, :, k] = np.transpose(rotation_matrix_b2w[:, :, k])
            
            # relative_target_position = np.subtract(self.target_position[:, k], self.X_states[0 : 3, k])
            # relative_target_position_BFrame = np.matmul(rotation_matrix_w2b[:, :, k], relative_target_position)

            # reference_yaw[k, 0] = np.arctan2(relative_target_position_BFrame[1], relative_target_position_BFrame[0])

            reference_yaw[k, 0] = np.arctan2(self.target_position[1, k] - self.X_states[1, k], self.target_position[0, k] - self.X_states[0, k])

        plt.plot(self.time_vec_state, np.degrees(reference_yaw), label='Relative Yaw, UAV-Target')
        plt.legend()
        plt.ylabel('Angle(deg)')
        plt.xlabel('Time (s)')
        plt.title("Yaw/Yaw Ref Comparison")

    def fun_plot_uav_speed(self):
        plt.figure("speed")
        vel_vec = []
        for k in range(0,self.time_steps):
            aux = math.sqrt((self.X_states[3, k] ** 2) + (self.X_states[4, k] ** 2) + (self.X_states[5, k] ** 2))
            vel_vec.append(aux)
        plt.plot(self.time_vec_state, vel_vec)
        plt.ylabel('speed (m/s)')
        plt.xlabel('Time (s)')
        plt.title("UAV speed")

    def fun_plot_uav_vxvyvz(self):
        plt.figure("velocity")
        # plt.plot(self.time_vec_state, self.X_states[3, :], label='v_x')
        # plt.plot(self.time_vec_state, self.X_states[4, :], label='v_y')
        # plt.plot(self.time_vec_state, self.X_states[5, :], label='v_z')
        plt.plot(self.time_vec_state, self.target_speed[0, :], label='v_x_t')
        plt.plot(self.time_vec_state, self.target_speed[1, :], label='v_y_t')
        plt.legend()
        plt.ylabel('v_x, v_y, v_z (m/s)')
        plt.xlabel('Time (s)')
        plt.title("UAV v_x, v_y, v_z coordinates")

    def fun_plot_obstacle_safety(self):
        plt.figure("obstacles distance")
        plt.axhline(y=self.safety_radius, color='b', linestyle=':', label="Safety Radius") 
        plt.axhline(y=self.obs_radius, color='r', linestyle=':', label="Obstacle Radius") 
        dist_vec = np.empty((3, self.time_steps))
        for k in range(0,self.time_steps):
            current_uav_pos = [self.X_states[0, k], self.X_states[1, k], self.X_states[2, k]]
            for i in range(0, self.obstacles_positions.shape[1]):
                dist = math.dist(current_uav_pos, self.obstacles_positions[:, i])
                dist_vec[i, k] = dist

        for i in range(0, self.obstacles_positions.shape[1]):
            plt.plot(self.time_vec_state, dist_vec[i, :], label=("Dist to obst " + str(i + 1)))
        plt.legend()
        plt.ylabel('Distance (m)')
        plt.xlabel('Time (s)')
        plt.title("UAV distance to obstacles")

    def fun_plot_occlusion(self):
        rotation_matrix_b2w = np.zeros((3, 3, self.time_steps))
        camera_rotation_matrix_b2c = np.zeros((3, 3, self.time_steps))
        rotation_matrix_w2b = np.zeros((3, 3, self.time_steps))            

        plot_arrays = np.zeros((self.obstacles_positions.shape[1], self.time_steps))
        a1 = np.zeros((self.obstacles_positions.shape[1], self.time_steps))
        iteration = 1

        for uav_i in range(0, self.uav_num * 9, 9):
            # Set up b2w and b2c rotation matrices throughout the horizon
            for k in range(0,self.time_steps):

                rotation_matrix_b2w[0, 0, k] = cos(self.X_states[8 + uav_i, k]) * cos(self.X_states[7 + uav_i, k])
                rotation_matrix_b2w[0, 1, k] = cos(self.X_states[8 + uav_i, k]) * sin(self.X_states[7 + uav_i, k]) * sin(self.X_states[6 + uav_i, k]) - sin(self.X_states[8 + uav_i, k]) * cos(self.X_states[6 + uav_i, k])
                rotation_matrix_b2w[0, 2, k] = cos(self.X_states[8 + uav_i, k]) * sin(self.X_states[7 + uav_i, k]) * cos(self.X_states[6 + uav_i, k]) + sin(self.X_states[8 + uav_i, k]) * sin(self.X_states[6 + uav_i, k])

                rotation_matrix_b2w[1, 0, k] = sin(self.X_states[8 + uav_i, k]) * cos(self.X_states[7 + uav_i, k])
                rotation_matrix_b2w[1, 1, k] = sin(self.X_states[8 + uav_i, k]) * sin(self.X_states[7 + uav_i, k]) * sin(self.X_states[6 + uav_i, k]) + cos(self.X_states[8 + uav_i, k]) * cos(self.X_states[6 + uav_i, k])
                rotation_matrix_b2w[1, 2, k] = sin(self.X_states[8 + uav_i, k]) * sin(self.X_states[7 + uav_i, k]) * cos(self.X_states[6 + uav_i, k]) - cos(self.X_states[8 + uav_i, k]) * sin(self.X_states[6 + uav_i, k])

                rotation_matrix_b2w[2, 0, k] = -sin(self.X_states[7 + uav_i, k])
                rotation_matrix_b2w[2, 1, k] = cos(self.X_states[7 + uav_i, k]) * sin(self.X_states[6 + uav_i, k])
                rotation_matrix_b2w[2, 2, k] = cos(self.X_states[7 + uav_i, k]) * cos(self.X_states[6 + uav_i, k])

                camera_rotation_matrix_b2c[0, 0, k] = cos(self.z_rotation_angle)
                camera_rotation_matrix_b2c[0, 1, k] = 0
                camera_rotation_matrix_b2c[0, 2, k] = sin(self.z_rotation_angle)

                camera_rotation_matrix_b2c[1, 0, k] = 0
                camera_rotation_matrix_b2c[1, 1, k] = 1
                camera_rotation_matrix_b2c[1, 2, k] = 0

                camera_rotation_matrix_b2c[2, 0, k] = -sin(self.z_rotation_angle)
                camera_rotation_matrix_b2c[2, 1, k] = 0
                camera_rotation_matrix_b2c[2, 2, k] = cos(self.z_rotation_angle)

            # Set up w2b rotation matrix throughout the horizon
            for k in range(0, self.time_steps):
                rotation_matrix_w2b[:, :, k] = np.transpose(rotation_matrix_b2w[:, :, k])

            for k in range(0,self.time_steps):
                relative_target_position = np.subtract(self.target_position[:, k], self.X_states[0+uav_i : 3+uav_i, k])
                relative_target_position_BFrame = np.matmul(rotation_matrix_w2b[:, :, k], relative_target_position)
                cam_pos = [0.226, 0, -0.089]
                relative_target_position_BFrame = np.subtract(relative_target_position_BFrame, cam_pos)
                relative_target_position_CFrame = np.matmul(camera_rotation_matrix_b2c[:, :, k], relative_target_position_BFrame)

                relative_target_position_CFrame_norm = np.linalg.norm(relative_target_position_CFrame)
                beta_versor = (1/relative_target_position_CFrame_norm) * relative_target_position_CFrame

                z_camera = [0, 0, 1]
                beta_angle = np.inner(beta_versor, z_camera)

                for i in range(0, self.obstacles_positions.shape[1]):
                    relative_obs_position = np.subtract(self.obstacles_positions[:, i], self.X_states[0+uav_i : 3+uav_i, k])
                    relative_obs_position_BFrame = np.matmul(rotation_matrix_w2b[:, :, k], relative_obs_position)
                    cam_pos = [0.226, 0, -0.089]
                    relative_obs_position_BFrame = np.subtract(relative_obs_position_BFrame, cam_pos)
                    relative_obs_position_CFrame = np.matmul(camera_rotation_matrix_b2c[:, :, k], relative_obs_position_BFrame)
                    relative_obs_position_CFrame_norm = np.linalg.norm(relative_obs_position_CFrame)
                    beta_versor_obs = (1/relative_obs_position_CFrame_norm) * relative_obs_position_CFrame
                    z_camera = [0, 0, 1]
                    
                    distance = np.linalg.norm(relative_target_position)
                    rel_dist = np.linalg.norm(relative_obs_position)

                    beta_angle_obs = np.inner(beta_versor_obs, z_camera)
                    ang1 = np.arccos(beta_angle)
                    ang2 = np.arccos(beta_angle_obs)
                    plot_arrays[i, k] = np.linalg.norm(beta_versor - beta_versor_obs)#np.degrees(abs(ang1-ang2))
                    # a1[i, k] = (self.obs_radius ** 2) / (4 * ((relative_obs_position[2] ** 2) - (self.obs_radius ** 2)))
                    a1[i, k] = (np.maximum(distance - rel_dist, 0) / (distance - rel_dist) ) * (self.obs_radius ** 2) / (4 * ((relative_obs_position[2] ** 2) - (self.obs_radius ** 2)))
        # fig, axs = plt.subplots(self.obstacles_positions.shape[1])
        # fig.suptitle('Target Occlusion')

        # axs[0].plot(self.time_vec_state, plot_arrays[0, :], label='Angle')
        # axs[0].axhline(y=0, color='b', linestyle=':', label='Desired value (fov center)')
        # axs[0].legend()
        # axs[0].set(ylabel='Angle(deg)')
        # axs[0].set_title("Angle between Target and Camera Center")

        # axs[1].plot(self.time_vec_state, plot_arrays[1, :], label='Horizontal Angle')
        # axs[1].axhline(y=horz_limit, color='r', linestyle=':', label='FOV Limit')
        # axs[1].legend()
        # axs[1].set(ylabel='Angle(deg)')

        # axs[2].plot(self.time_vec_state, plot_arrays[2, :], label='Vertical Angle')
        # axs[2].axhline(y=vert_limit, color='r', linestyle=':', label='FOV Limit')
        # axs[2].legend()
        # axs[2].set(ylabel='Angle(deg)', xlabel='Time (s)')

        plt.figure("occlusion")
        colors = ['r', 'b', 'g']
        for i in range(0, self.obstacles_positions.shape[1]):
            plt.plot(self.time_vec_state, plot_arrays[i, :], label='obstacle ' + str(i + 1), color = colors[i])
            plt.plot(self.time_vec_state, a1[i, :], label = 'a1 obs ' + str(i + 1), linestyle = '--', color = colors[i])
        plt.legend()
        plt.ylabel('Angle(rad)')
        plt.xlabel('Time (s)')
        plt.title("Target Occlusion")


    def plot_show(self):
        plt.show(block=False)
        input()
        #plt.close("all")

        #self.plotting_3d()


def main():
    plotter = plot_graphs_class()

if __name__ == "__main__":
    main()  