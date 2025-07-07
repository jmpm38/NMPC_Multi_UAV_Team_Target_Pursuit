import casadi as ca
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from casadi import sin, cos, pi, tan
import math
import yaml
from scipy.interpolate import interp1d
import scienceplots


# Right now the class needs to be initialized -> set_variables (receive target positions, UAV states, UAV control inputs) -> plot_starter (calls plot3D after input())

class plot_graphs_class:

    def __init__(self):
        plt.style.use('science')
        # plt.rcParams['text.usetex'] = False
        self.plot_sizerino = (7, 5)
        self.plot_sizerino_2 = (7, 7)
        self.yaml_handling()
        self.first = True
        self.real_uav_num = self.uav_num

    def plot_starter(self):
        self.fun_plot_uav_target_XY()
        self.fun_plot_uav_xyz()
        self.fun_plot_uav_target_dist()
        self.fun_plot_uav_roll_roll_ref()
        self.fun_plot_uav_pitch_pitch_ref()
        self.fun_plot_uav_yaw_yaw_ref()
        # self.plot_slack_variables()
        self.plot_alpha_vs_desired_alpha()
        # self.fun_plot_uav_speed()
        # if(self.target_speed != 0):
        self.fun_plot_uav_vxvyvz()
        self.fun_plot_target_vxvy()

        # OBSTACLES:
        # self.fun_plot_obstacle_safety()
        # self.fun_plot_occlusion()

        self.fun_plot_fov()
        self.fun_plot_input_constr()

        self.set_uavs_positions(self.X_states, self.uav_i)

        if(self.uav_num>1):
            self.fun_plot_distance_between_uavs()
            self.fun_plot_angular_separations_uavs()

        if((self.uav_i == (self.real_uav_num - 1)) and self.real_uav_num != 1):
            self.plot_angular_separation()
            self.plot_uavs_distance()

        # show plots
        self.plot_show()

    def yaml_handling(self):
            # Read the YAML file
            with open('src/multi_drone/config/config.yaml', 'r') as yaml_file:
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

            self.uav_num = UAVs_Initial_State["uav_num"]

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

    def set_variables(self, target_positions, X_states, U_controls, horizon, time_vec, slack_variables, alpha_ref, uav_i, target_speed=0):
        self.N_horizon = horizon
        # if(self.first):
        #     self.uav_states = np.zeros((X_states.shape[0], X_states.shape[1], self.real_uav_num))
        #     self.desired_alphas = np.zeros((X_states.shape[1], self.real_uav_num))
        # self.first = False
        self.uav_num = 1

        self.target_position = target_positions # 3, self.time_steps + 1
        self.target_speed = target_speed #2, self.time_steps + 1
        self.X_states = X_states # 9*self.uav_num , self.time_steps + 1
        self.U_controls = U_controls # 4*self.uav_num , self.time_steps
        self.time_vec_state = time_vec
        self.time_vec_control = time_vec
        self.time_steps = self.X_states.shape[1]
        self.time_vec_target = time_vec
        self.time_vec = time_vec
        self.slack_variables = slack_variables
        self.alpha_ref = alpha_ref
        self.uav_i = uav_i
        # self.set_desired_alpha(self.alpha_ref, self.uav_i)


    def fun_plot_uav_target_XY(self):
        plt.figure('xy_target', figsize=self.plot_sizerino)

        # Plot UAV positions
        i = 1
        for uav_i in range(0, self.uav_num * 9, 9):
            u = np.cos(self.X_states[8 + uav_i, :])
            v = np.sin(self.X_states[8 + uav_i, :])

            plt.scatter(self.X_states[0 + uav_i, :], self.X_states[1 + uav_i, :], label = rf'$p_{{{uav_i + 1}}}$', s=10)
            # plt.quiver(self.X_states[0 + uav_i, :], self.X_states[1 + uav_i, :], u, v, angles='xy', scale_units='xy', scale=0.5, color='red')
            i += 1

        # Plot target position
        # if(self.uav_i == 2):
        plt.scatter(self.target_position[0, :], self.target_position[1, :], color='r', label="$p_T$", marker="*", s=10)
        plt.grid()

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect = [0.05, 0.05, 1, 0.95])
        plt.ylabel('y (m)')
        plt.xlabel('x (m)')
        plt.title("UAV/Target Position on XY plane")

    def fun_plot_uav_xyz(self):
        i = 1
        #plt.figure("z")
        for uav_i in range(0, self.uav_num*9, 9):
            plt.figure('uav ' + str(i) + ' xyz', figsize=self.plot_sizerino)
            plt.plot(self.time_vec_state, self.X_states[0 + uav_i, :], label=('x'))
            plt.plot(self.time_vec_state, self.X_states[1 + uav_i, :], label=('y'))
            plt.plot(self.time_vec_state, self.X_states[2 + uav_i, :], label=('z'))
            plt.plot(self.time_vec_state, self.target_position[0, :], label=('$x_T$' + str(i)))
            # plt.plot(self.time_vec_state, self.target_position[1, :], label=('$y_T$' + str(i)))

            # plt.plot(self.time_vec_state, self.target_position[0, :], label=('X_t' + str(i)))
            # plt.plot(self.time_vec_state, self.target_position[1, :], label=('Y_t' + str(i)))
            # plt.plot(self.time_vec_state, self.target_position[2, :], label=('Z_t' + str(i)))
            plt.axhline(y=self.h_min, color='r', linestyle='--')
            # plt.axhline(y=30, color='r', linestyle='--')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout(rect = [0.05, 0.05, 1, 0.95])
            plt.ylabel('Position (m)')
            plt.xlabel('Time (s)')
            plt.title("UAV Position Over Time")
            plt.grid()
            i = i + 1

    def fun_plot_uav_target_dist(self):
        plt.figure("distance to target", figsize=self.plot_sizerino)
        i = 1
        # plt.axhline(y=self.Standoff_distance, color='b', linestyle=':', label="Desired distance to Target")
        for uav_i in range(0, self.uav_num*9, 9):
            dist_vec = []
            for k in range(0,self.time_steps):
                current_uav_pos = [self.X_states[0+uav_i, k], self.X_states[1+uav_i, k], self.X_states[2+uav_i, k]]
                current_target_pos = [self.target_position[0, k], self.target_position[1, k], self.target_position[2, k]]
                dist = math.dist(current_uav_pos, current_target_pos)
                dist_vec.append(dist)
            plt.plot(self.time_vec_state, dist_vec, label='$||p_{T} - p_{UAV}||$')#, label=("UAV " + str(i) + " distance to the target"))
            i = i + 1
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        # plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        plt.ylabel('Distance (m)')
        plt.xlabel('Time (s)')
        plt.title("Distance UAV - Target")
        plt.grid()


    def fun_plot_input_constr(self):
        i = 1
        for uav_i in range(0, self.uav_num*4, 4):
            fig, axs = plt.subplots(4, figsize=self.plot_sizerino_2)
            fig.suptitle('UAV Control Inputs')
            fig.canvas.manager.set_window_title('Inputs')
            axs[0].plot(self.time_vec_control, self.U_controls[0+uav_i, :], label='$T$')
            # axs[0].axhline(y=self.T_min, color='r', linestyle=':', label="Thrust boundaries")
            # axs[0].axhline(y=self.T_max, color='r', linestyle=':')
            # axs[0].axhline(y=self.U_ref[0], color='b', linestyle=':', label="Thrust reference")
            axs[0].axhline(y=0, color='r', linestyle=':', label="$T_{max/min}$")
            axs[0].axhline(y=1, color='r', linestyle=':')
            axs[0].axhline(y=0.5433419696, color='b', linestyle=':', label="$T_{hover}$")
            axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
            # axs[0].set_title('Throttle')
            axs[0].set(ylabel='Throttle')
            axs[0].grid()

            axs[1].plot(self.time_vec_control[2:], self.U_controls[1+uav_i, 2:], label='$\phi_{ref}$')
            axs[1].axhline(y=-self.pitch_max, color='r', linestyle=':', label="$\phi_{max/min}$")
            axs[1].axhline(y=self.pitch_max, color='r', linestyle=':')
            axs[1].axhline(y=self.U_ref[1], color='b', linestyle=':', label="$\phi_{des}$")
            axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
            axs[1].grid()
            # axs[1].set_title('Roll Angle')
            axs[1].set(ylabel='Angle (rad)')

            axs[2].plot(self.time_vec_control[2:], self.U_controls[2+uav_i, 2:], label=r'$\theta_{ref}$')
            axs[2].axhline(y=-self.roll_max, color='r', linestyle=':', label=r"$\theta_{max/min}$")
            axs[2].axhline(y=self.roll_max, color='r', linestyle=':')
            axs[2].axhline(y=self.U_ref[2], color='b', linestyle=':', label=r"$\theta_{des}$")
            axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))
            axs[2].grid()
            # axs[2].set_title('Pitch Angle')
            axs[2].set(ylabel='Angle (rad)')

            discontinuity_threshold = 1.5
            mask = np.abs(np.diff(self.U_controls[3+uav_i, 2:], axis=0)) > discontinuity_threshold
            yaw_to_plot = np.insert(self.U_controls[3+uav_i, 2:], np.where(mask)[0] + 1, np.nan)
            time_vec_aux = np.insert(self.time_vec, np.where(mask)[0] + 1, np.nan)
            axs[3].plot(time_vec_aux[2:], yaw_to_plot, label='$\psi_{ref}$')
            axs[3].axhline(y=-pi, color='r', linestyle=':')
            axs[3].axhline(y=pi, color='r', linestyle=':', label='$\pm\pi$')
            # axs[3].set_title('Yaw Angle')
            axs[3].set(ylabel='Angle (rad)', xlabel='Time (s)')
            axs[3].grid()
            axs[3].legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout(rect=[0, 0, 1, 0.95])
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
                # print(self.time_vec[k])
                # print(relative_target_position_BFrame)
                cam_pos = [0.226, 0, -0.089]
                relative_target_position_BFrame = np.subtract(relative_target_position_BFrame, cam_pos)
                relative_target_position_CFrame = np.matmul(camera_rotation_matrix_b2c[:, :, k], relative_target_position_BFrame)
                # print(relative_target_position_CFrame)

                relative_target_position_CFrame_norm = np.linalg.norm(relative_target_position_CFrame)
                beta_versor = (1/relative_target_position_CFrame_norm) * relative_target_position_CFrame
                # print(beta_versor)

                z_camera = [0, 0, 1]
                beta_angle = np.inner(beta_versor, z_camera)
                # print(beta_angle)
                # print(np.degrees(np.arccos(beta_angle)))
                # if((k>20) and (np.degrees(np.arccos(beta_angle))>40)):
                #     plot_arrays[0, k] = plot_arrays[0, k-1]
                #     plot_arrays[1, k] = plot_arrays[1, k-1]
                #     plot_arrays[2, k] = plot_arrays[2, k-1]
                #     continue
                plot_arrays[0, k] = np.degrees(np.arccos(beta_angle))
                
                # print("----------------------------")
                # print(self.time_vec_state[k])
                # # print(self.X_states[6 + uav_i, k])
                # # print(self.X_states[7 + uav_i, k])
                # print(self.X_states[8 + uav_i, k])
                # print(relative_target_position)
                # print(relative_target_position_BFrame)
                # print(relative_target_position_CFrame)
                # # print(relative_target_position_CFrame_norm)
                # # print(beta_versor)
                # # print(beta_angle)
                # # print(plot_arrays[0, k])
                # if(plot_arrays[0, k] > 100):
                #     print("maior que 100")
                #     print("*********************************************************************************")
                #     input()

                horizontal = np.array([beta_versor[0], 0, beta_versor[2]])
                vertical = np.array([0, beta_versor[1], beta_versor[2]])

                horizontal_norm = np.linalg.norm(horizontal)
                vertical_norm = np.linalg.norm(vertical)

                horizontal = (1/horizontal_norm) * horizontal
                vertical = (1/vertical_norm) * vertical

                horz_angle = np.inner(horizontal, z_camera)
                vert_angle = np.inner(vertical, z_camera)
                
                plot_arrays[1, k] = np.degrees(np.arccos(horz_angle))
                plot_arrays[2, k] = np.degrees(np.arccos(vert_angle))
                # print(np.degrees(np.arccos(horz_angle)))
                # print(np.degrees(np.arccos(vert_angle)))
                # input()

                if(beta_versor[0] < 0):
                    plot_arrays[1, k] = - plot_arrays[1, k]
                if(beta_versor[1] < 0):
                    plot_arrays[2, k] = - plot_arrays[2, k]

                # horz_limit = cos(self.alpha_h/2)
                # vert_limit = cos(self.alpha_v/2)
                horz_limit = np.degrees(self.alpha_h/2)
                vert_limit = np.degrees(self.alpha_v/2)

                # if(k>100 and plot_arrays[0,k]>40):
                #     plot_arrays[0,k] = plot_arrays[0,k-1]
                # if(k>100 and plot_arrays[1,k]>40):
                #     plot_arrays[1,k] = plot_arrays[1,k-1]
                # if(k>100 and plot_arrays[2,k]>40):
                #     plot_arrays[2,k] = plot_arrays[2,k-1]

            fig, axs = plt.subplots(3, figsize=self.plot_sizerino_2)
            fig.canvas.manager.set_window_title('FOV')
            fig.suptitle('Perception Objectives')

            axs[0].plot(self.time_vec_state, plot_arrays[0, :], label=r'$\beta$')
            axs[0].axhline(y=0, color='b', linestyle=':', label='Camera Center')
            axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
            axs[0].grid()
            axs[0].set(ylabel='Angle(deg)')
            # axs[0].set_title("Angle between Target and Camera Center")

            axs[1].plot(self.time_vec_state, plot_arrays[1, :], label=r'$\beta_h$')
            axs[1].axhline(y=horz_limit, color='r', linestyle=':', label=r'$\beta_{h, max}$')
            axs[1].axhline(y=-horz_limit, color='r', linestyle=':', label=r'$\beta_{h, max}$')
            axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
            axs[1].grid()
            axs[1].set(ylabel='Angle(deg)')

            axs[2].plot(self.time_vec_state, plot_arrays[2, :], label=r'$\beta_v$')
            axs[2].axhline(y=vert_limit, color='r', linestyle=':', label=r'$\beta_{v, max}$')
            axs[2].axhline(y=-vert_limit, color='r', linestyle=':', label=r'$\beta_{v, max}$')
            axs[2].grid()
            axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))
            axs[2].set(ylabel='Angle(deg)', xlabel='Time (s)')

            iteration = iteration + 1
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # Define camera FOV in degrees
            fov_horizontal = 69  # degrees
            fov_vertical = 42    # degrees

            x_deg_min = -(fov_horizontal / 2)
            x_deg_max = (fov_horizontal / 2)
            y_deg_min = -(fov_vertical / 2)
            y_deg_max = (fov_vertical / 2)

            # Define some margins around the FOV (for targets outside the FOV)
            margin_horizontal = 5  # degrees on each side
            margin_vertical = 3    # degrees on each side

            # Separate target positions into horizontal and vertical lists
            horizontal_angles = plot_arrays[1, :]
            vertical_angles = plot_arrays[2, :]

            # Define the grid ranges for the heatmap (FOV + margins)
            x_range = np.linspace(-(fov_horizontal/2 + margin_horizontal), 
                                (fov_horizontal/2 + margin_horizontal), 26)  # 101 to match edges
            y_range = np.linspace(-(fov_vertical/2 + margin_vertical), 
                                (fov_vertical/2 + margin_vertical), 26)      # 101 to match edges

            # Create a 2D histogram of the target positions
            heatmap, xedges, yedges = np.histogram2d(horizontal_angles, vertical_angles, 
                                                    bins=[x_range, y_range])

            # Generate the center points of each bin for plotting
            X, Y = np.meshgrid(xedges[:-1] + np.diff(xedges)/2, yedges[:-1] + np.diff(yedges)/2)

            # Plot the heatmap
            plt.figure("FOV_heatmap", figsize=(8, 6))
            plt.contourf(X, Y, heatmap.T, levels=30, cmap='hot')
            plt.colorbar(label='Target Density')

            # Set axis labels
            plt.xlabel('Horizontal Angle (degrees)')
            plt.ylabel('Vertical Angle (degrees)')

            # Add lines representing the camera's actual FOV
            plt.plot([x_deg_min, x_deg_max], [y_deg_min, y_deg_min], color='white', linewidth=2, linestyle='--')  # Bottom boundary
            plt.plot([x_deg_min, x_deg_max], [y_deg_max, y_deg_max], color='white', linewidth=2, linestyle='--')  # Top boundary
            plt.plot([x_deg_min, x_deg_min], [y_deg_min, y_deg_max], color='white', linewidth=2, linestyle='--')  # Left boundary
            plt.plot([x_deg_max, x_deg_max], [y_deg_min, y_deg_max], color='white', linewidth=2, linestyle='--')  # Right boundary

            # Title
            plt.title('Camera FOV Heatmap with Target Positions')


    #TODO:
    def fun_plot_distance_between_uavs(self):
        plt.figure("Distance between UAVs", figsize=self.plot_sizerino)
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
        plt.figure("Angular distance between UAVs", figsize=self.plot_sizerino)
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
        plt.figure("roll", figsize=self.plot_sizerino)
        plt.plot(self.time_vec_state[2:], np.degrees(self.X_states[6, 2:]), label='$\phi$')
        plt.plot(self.time_vec_control[2:], np.degrees(self.U_controls[1, 2:]), label='$\phi_{ref}$')
        plt.axhline(y=np.degrees(self.roll_max), color='r', linestyle='--', label='$\phi_{max/min}$')
        plt.axhline(y=-np.degrees(self.roll_max), color='r', linestyle='--')
        plt.axhline(y=self.U_ref[1], color='b', linestyle=':', label='$\phi_{des}$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect = [0.05, 0.05, 1, 0.95])
        plt.ylabel('Angle(deg)')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.title("Roll Over Time")

    def fun_plot_uav_pitch_pitch_ref(self):
        plt.figure("pitch", figsize=self.plot_sizerino)
        plt.plot(self.time_vec_state[2:], np.degrees(self.X_states[7, 2:]), label=r'$\theta$')
        plt.plot(self.time_vec_control[2:], np.degrees(self.U_controls[2, 2:]), label=r'$\theta_{ref}$')
        plt.axhline(y=np.degrees(self.pitch_max), color='r', linestyle='--', label=r'$\theta_{max/min}$')
        plt.axhline(y=-np.degrees(self.pitch_max), color='r', linestyle='--')
        plt.axhline(y=self.U_ref[2], color='b', linestyle=':', label = r'$\theta_{des}$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect = [0.05, 0.05, 1, 0.95])
        plt.ylabel('Angle(deg)')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.title("Pitch Over Time")

    def fun_plot_uav_yaw_yaw_ref(self):
        plt.figure("yaw", figsize=self.plot_sizerino)

        discontinuity_threshold = 360 
        mask = np.abs(np.diff(np.degrees(self.X_states[8, 2:]), axis=0)) > discontinuity_threshold
        yaw_to_plot = np.insert(np.degrees(self.X_states[8, 2:]), np.where(mask)[0] + 1, np.nan)
        time_vec_aux = np.insert(self.time_vec, np.where(mask)[0] + 1, np.nan)
        plt.plot(time_vec_aux[2:], yaw_to_plot, label='$\psi$')

        discontinuity_threshold = 360
        mask = np.abs(np.diff(self.U_controls[3, 2:], axis=0)) > discontinuity_threshold
        yaw_to_plot = np.insert(self.U_controls[3, 2:], np.where(mask)[0] + 1, np.nan)
        time_vec_aux = np.insert(self.time_vec, np.where(mask)[0] + 1, np.nan)
        plt.plot(time_vec_aux[2:], np.degrees(yaw_to_plot), label='$\psi_{ref}$')
        reference_yaw = np.zeros((self.time_steps, 1))
        for k in range(0,self.time_steps):
            reference_yaw[k, 0] = np.arctan2(self.target_position[1, k] - self.X_states[1, k], self.target_position[0, k] - self.X_states[0, k])
            reference_yaw[k, 0] = np.rad2deg(reference_yaw[k, 0])

        discontinuity_threshold = 360
        mask = np.abs(np.diff(reference_yaw, axis=0)) > discontinuity_threshold
        yaw_to_plot = np.insert(reference_yaw, np.where(mask)[0] + 1, np.nan)
        time_vec_aux = np.insert(self.time_vec, np.where(mask)[0] + 1, np.nan)
        plt.plot(time_vec_aux, yaw_to_plot, label='$\psi_{des}$')
        plt.axhline(y=-180, color='r', linestyle=':')
        plt.axhline(y=180, color='r', linestyle=':', label='$\pm\pi$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect = [0.05, 0.05, 1, 0.95])
        plt.ylabel('Angle(deg)')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.title("Yaw Over Time")

    def fun_plot_uav_speed(self):
        plt.figure("speed", figsize=self.plot_sizerino)
        vel_vec = []
        for k in range(0,self.time_steps):
            aux = math.sqrt((self.X_states[3, k] ** 2) + (self.X_states[4, k] ** 2) + (self.X_states[5, k] ** 2))
            vel_vec.append(aux)
        plt.plot(self.time_vec_state, vel_vec)
        plt.ylabel('speed (m/s)')
        plt.xlabel('Time (s)')
        plt.title("UAV speed")

    def fun_plot_uav_vxvyvz(self):
        plt.figure("velocity", figsize=self.plot_sizerino)
        plt.plot(self.time_vec_state, self.X_states[3, :], label='$v_x$')
        plt.plot(self.time_vec_state, self.X_states[4, :], label='$v_y$')
        plt.plot(self.time_vec_state, self.X_states[5, :], label='$v_z$')
        # plt.plot(self.time_vec_state, self.target_speed[0, :], label='v_x_t')
        # plt.plot(self.time_vec_state, self.target_speed[1, :], label='v_y_t')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect = [0.05, 0.05, 1, 0.95])
        plt.ylabel('Velocity (m/s)')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.title("UAV Velocity Over Time")

    def fun_plot_target_vxvy(self):
        plt.figure("velocity_target", figsize=self.plot_sizerino)
        plt.plot(self.time_vec_state, self.target_speed[0, :], label='$v_{x,T}$')
        plt.plot(self.time_vec_state, self.target_speed[1, :], label='$v_{y,T}$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect = [0.05, 0.05, 1, 0.95])
        plt.ylabel('Velocity (m/s)')
        plt.xlabel('Time (s)')
        plt.title("Target $v_{x,T}$, $v_{y,T}$ coordinates")
        plt.grid()

    def fun_plot_obstacle_safety(self):
        plt.figure("obstacles distance", figsize=self.plot_sizerino)
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
        if(self.uav_i == (self.real_uav_num - 1)):
            pass
        input()
        # plt.close("all")

        #self.plotting_3d()

    ###########################################################
    # New part: alphas and slack variables

    def plot_slack_variables(self):
        plt.figure('slack variables', figsize=self.plot_sizerino)
        # plt.plot(self.time_vec, self.slack_variables[0, :], label=('$S_1$'))
        # plt.plot(self.time_vec, self.slack_variables[1, :], label=('$S_2$'))
        # plt.plot(self.time_vec, self.slack_variables[2, :], label=('$S_3$'))
        plt.plot(self.time_vec, self.slack_variables[1, :], label = rf'$S_2,\, uav_{{{self.uav_i + 1}}}$')
        plt.plot(self.time_vec, self.slack_variables[2, :], label = rf'$S_3,\, uav_{{{self.uav_i + 1}}}$')
        plt.axhline(y=0, color='r', linestyle=':')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect = [0.05, 0.05, 1, 0.95])
        plt.ylabel('Slack Variable')
        plt.xlabel('Time (s)')
        plt.title("Slack Variables Over Time")
        plt.grid()

    def plot_alpha_vs_desired_alpha(self):
        alpha_angle = np.zeros((self.alpha_ref.shape))
        for k in range(self.time_steps):
            current_uav_pos = ca.vertcat(self.X_states[0, k], self.X_states[1, k], self.X_states[2, k])
            alpha_angle[0, k] = np.arctan2(self.target_position[1, k] - current_uav_pos[1], self.target_position[0, k] - current_uav_pos[0])
        alpha_ref = np.transpose(self.alpha_ref)
        alpha_angle = np.transpose(alpha_angle)
        self.alpha_angle_now = alpha_angle
        plt.figure('Alpha Angle', figsize=self.plot_sizerino)
        num = self.uav_i + 1

        discontinuity_threshold = 2
        mask = np.abs(np.diff(alpha_angle, axis=0)) > discontinuity_threshold
        alpha_angle = np.insert(alpha_angle, np.where(mask)[0] + 1, np.nan)
        time_vec_aux = np.insert(self.time_vec, np.where(mask)[0] + 1, np.nan)
        plt.plot(time_vec_aux, alpha_angle, label=rf'$\alpha_{{{num}}}$')

        discontinuity_threshold = 2
        mask = np.abs(np.diff(alpha_ref, axis=0)) > discontinuity_threshold
        mask2 = self.time_vec < 7
        alpha_ref = np.insert(alpha_ref, np.where(mask)[0] + 1, np.nan)
        alpha_ref = np.insert(alpha_ref, np.where(mask2)[0] + 1, np.nan)
        time_vec_aux = np.insert(self.time_vec, np.where(mask)[0] + 1, np.nan)
        time_vec_aux = np.insert(time_vec_aux, np.where(mask2)[0] + 1, np.nan)
        plt.plot(time_vec_aux, alpha_ref, label=rf'$\alpha_{{\text{{ref}}, {num}}}$')

        plt.axhline(np.pi, linestyle = ':', color='r')
        plt.axhline(-np.pi, linestyle = ':', color='r')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        plt.ylabel('Alpha (rad)')
        plt.xlabel('Time (s)')
        plt.title("Alpha Angle")
        plt.grid()




    def set_uavs_positions(self, uav_i_states, uav_i):
        if(uav_i == 0):
            self.uav1 = uav_i_states
            self.alpha_ref_1 = self.alpha_ref
            self.alpha_angle_1 = self.alpha_angle_now
            print(self.uav1.shape)
        if(uav_i == 1):
            self.uav2 = uav_i_states
            self.alpha_ref_2 = self.alpha_ref
            self.alpha_angle_2 = self.alpha_angle_now
            print(self.uav2.shape)
        if(uav_i == 2):
            self.uav3 = uav_i_states
            self.alpha_ref_3 = self.alpha_ref
            self.alpha_angle_3 = self.alpha_angle_now
            print(self.uav3.shape)
            # List of arrays
            arrays = [self.uav1, self.uav2, self.uav3]

            # Step 1: Determine the largest number of columns
            max_columns = min(arr.shape[1] for arr in arrays)

            # Step 2: Interpolate each array to have the same number of columns
            interpolated_arrays = []
            for arr in arrays:
                original_columns = arr.shape[1]
                x_original = np.linspace(0, 1, original_columns)  # Original points
                x_new = np.linspace(0, 1, max_columns)            # New points for interpolation
                interpolated_array = np.zeros((arr.shape[0], max_columns))  # Initialize new array

                for i in range(arr.shape[0]):
                    interp_func = interp1d(x_original, arr[i, :], kind='linear')
                    interpolated_array[i, :] = interp_func(x_new)

                interpolated_arrays.append(interpolated_array)

            self.uav_states = np.zeros((interpolated_arrays[0].shape[0], interpolated_arrays[0].shape[1], self.real_uav_num))
            for uav_i in range(self.real_uav_num):
                self.uav_states[:, :, uav_i] = interpolated_array[uav_i]
            print(self.uav_states.shape)

    # def set_desired_alpha(self, alpha_vector, uav_i):
    #     self.desired_alphas[:, uav_i] = alpha_vector


    def plot_angular_separation(self):
        self.separation_angles = np.zeros((3, self.time_steps))
        self.ref_separation_angles = np.zeros((3, self.time_steps))
        # for k in range(self.time_steps):
        #     iterator = 0
        #     for uav_i in range(0, self.real_uav_num, 1):
        #         uav_i_pos = ca.vertcat(self.uav_states[0, k, uav_i], self.uav_states[1, k, uav_i], self.uav_states[2, k, uav_i])
        #         alpha_angle_i = np.arctan2(self.target_position[1, k] - uav_i_pos[1], self.target_position[0, k] - uav_i_pos[0])
        #         for uav_j in range(uav_i + 1, self.real_uav_num, 1):
        #             uav_j_pos = ca.vertcat(self.uav_states[0, k, uav_j], self.uav_states[1, k, uav_j], self.uav_states[2, k, uav_j])
        #             alpha_angle_j = np.arctan2(self.target_position[1, k] - uav_j_pos[1], self.target_position[0, k] - uav_j_pos[0])
        #             self.separation_angles[iterator, k] = (np.degrees(alpha_angle_i - alpha_angle_j) + 180) % 360 - 180
        #             iterator = iterator + 1
        #             # self.separation_angles[uav_i, uav_j] = np.abs(np.degrees(alpha_angle_i - alpha_angle_j))
        for k in range(self.time_steps):
            try:
                self.separation_angles[0, k] = ((np.degrees(self.alpha_angle_1[k, 0] - self.alpha_angle_2[k, 0]) + 180) % 360) - 180
                self.ref_separation_angles[0, k] = (np.degrees(self.alpha_ref_1[0, k] - self.alpha_ref_2[0, k]) + 180) % 360 - 180
            except:
                pass
            try:
                self.separation_angles[1, k] = ((np.degrees(self.alpha_angle_1[k, 0] - self.alpha_angle_3[k, 0]) + 180) % 360 )- 180
                self.ref_separation_angles[1, k] = (np.degrees(self.alpha_ref_1[0, k] - self.alpha_ref_3[0, k]) + 180) % 360 - 180
            except:
                pass
            try:
                self.separation_angles[2, k] = ((np.degrees(self.alpha_angle_2[k, 0] - self.alpha_angle_3[k, 0]) + 180) % 360) - 180
                self.ref_separation_angles[2, k] = (np.degrees(self.alpha_ref_2[0, k] - self.alpha_ref_3[0, k]) + 180) % 360 - 180
            except:
                pass

            # if((np.abs(self.separation_angles[0,k]) < 30 )and (k>100)):
            #     self.separation_angles[0,k] = self.separation_angles[0,k-1]
            # if((np.abs(self.separation_angles[1,k]) < 30 )and (k>100)):
            #     self.separation_angles[1,k] = self.separation_angles[1,k-1]
            # if((np.abs(self.separation_angles[2,k]) < 30 )and (k>100)):
            #     self.separation_angles[2,k] = self.separation_angles[2,k-1]


        plt.figure('Angular Separation', figsize=self.plot_sizerino)
        iterator = 0
        for uav_i in range(0, self.real_uav_num, 1):
            for uav_j in range(uav_i + 1, self.real_uav_num, 1):
                # plt.plot(self.time_vec, self.separation_angles[iterator, :], label=('angular distance, uavs ' + str(uav_i + 1) + ',' + str(uav_j + 1)))
                discontinuity_threshold = 180 
                mask = np.abs(np.diff(self.separation_angles[iterator, :], axis=0)) > discontinuity_threshold
                yaw_to_plot = np.insert(self.separation_angles[iterator, :], np.where(mask)[0] + 1, np.nan)
                time_vec_aux = np.insert(self.time_vec, np.where(mask)[0] + 1, np.nan)
                plt.plot(time_vec_aux, yaw_to_plot, label = rf'$\text{{angle}}_{{\text{{wrapped}}}} (\alpha_{{{uav_i + 1}}} - \alpha_{{{uav_j + 1}}})$')
                # plt.plot(self.time_vec, self.ref_separation_angles[iterator, :], label = ('ref angular distance, uavs ' + str(uav_i + 1) + ',' + str(uav_j + 1)))
                iterator = iterator + 1
        plt.axhline(y=0, color='r', linestyle=':', label = rf'$(60k)^\circ,\, -3 \leq k \leq 3,\, k \in \mathbb{{Z}}$')
        plt.axhline(y=60, color='r', linestyle=':')
        plt.axhline(y=-60, color='r', linestyle=':')
        plt.axhline(y=120, color='r', linestyle=':')
        plt.axhline(y=-120, color='r', linestyle=':')
        plt.axhline(y=-180, color='r', linestyle=':')
        plt.axhline(y=180, color='r', linestyle=':')
        plt.ylabel('Angle (deg)')
        plt.xlabel('Time (s)')
        plt.title("Angular Separation between UAVs relative to Target")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        plt.grid()

        plt.figure('FIM', figsize=self.plot_sizerino)
        fim_values=np.zeros((self.time_steps, 1))
        for k in range(self.time_steps):
            for i in range(3):
                degreeees = np.deg2rad(self.separation_angles[i, k])
                # print(degreeees)
                # input()
                fim_values[k, 0] = fim_values[k, 0] + (np.sin(degreeees) ** 2)
                # print(np.sin(self.separation_angles[i, k]) ** 2)
                # print(fim_values[k,0])
                # input()

            # if((fim_values[k,0] < 1) and k>100):
            #     fim_values[k,0] = fim_values[k-1,0]

        plt.plot(self.time_vec, fim_values, label = r'$\det FIM$')
        plt.axhline(3*(np.sin(pi/3)**2), color='b', linestyle=':', label = r'$\det FIM_{max}$')
        # plt.ylabel('')
        plt.xlabel('Time (s)')
        plt.title("$\det$ FIM Numerator Over Time")
        plt.ylim((-0.2, 2.45))
        plt.grid()
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    def plot_solving_time(self, solving_times):
        plt.figure('Optimization Solving Time', figsize=self.plot_sizerino)
        # for k in range(solving_times.shape[1]):
        #     if((k>200) and (solving_times[1, k] > 0.3)):
        #         solving_times[1, k] = solving_times[1, k-1]
        plt.plot(solving_times[0, :] - solving_times[0, 0], 0.8*solving_times[1, :])#, marker = ':')
        plt.axhline(y=0.2, color='r', linestyle=':', label = '5 Hz Solving Time')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        plt.ylabel('Solving Time (ms)')
        plt.xlabel('Time (s)')
        plt.title("Solving Times")
        plt.grid()


    def plot_uavs_distance(self):
        plt.figure('uavs distance', figsize=self.plot_sizerino)

        self.distances = np.zeros((3, self.time_steps))
        for k in range(self.time_steps):
            try:
                uav_1_pos = ca.vertcat(self.uav1[0, k], self.uav1[1, k], self.uav1[2, k])
            except:
                pass
            try:
                uav_2_pos = ca.vertcat(self.uav2[0, k], self.uav2[1, k], self.uav2[2, k])
            except:
                pass
            try:
                uav_3_pos = ca.vertcat(self.uav3[0, k], self.uav3[1, k], self.uav3[2, k])
            except:
                pass
            try:
                self.distances[0, k] = np.linalg.norm(np.subtract(uav_1_pos,uav_2_pos))
            except:
                pass
            try:
                self.distances[1, k] = np.linalg.norm(np.subtract(uav_1_pos,uav_3_pos))
            except:
                pass
            try:
                self.distances[2, k] = np.linalg.norm(np.subtract(uav_2_pos,uav_3_pos))
            except:
                pass
            # iterator = 0
            # for uav_i in range(0, self.real_uav_num, 1):
            #     uav_i_pos = ca.vertcat(self.uav_states[0, k, uav_i], self.uav_states[1, k, uav_i], self.uav_states[2, k, uav_i])
            #     for uav_j in range(uav_i + 1, self.real_uav_num, 1):
            #         uav_j_pos = ca.vertcat(self.uav_states[0, k, uav_j], self.uav_states[1, k, uav_j], self.uav_states[2, k, uav_j])
            #         self.distances[iterator, k] = np.linalg.norm(uav_i_pos-uav_j_pos)
            #         iterator = iterator + 1
            #         # self.separation_angles[uav_i, uav_j] = np.abs(np.degrees(alpha_angle_i - alpha_angle_j))

        iterator = 0
        for uav_i in range(0, self.real_uav_num, 1):
            for uav_j in range(uav_i + 1, self.real_uav_num, 1):
                plt.plot(self.time_vec, self.distances[iterator, :], label = rf'$\|p_{{{uav_i + 1}}} - p_{{{uav_j + 1}}}\|$')
                iterator = iterator + 1
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        plt.ylabel('Distance (m)')
        plt.xlabel('Time (s)')
        plt.title("Distance between UAVs")
        plt.grid()

def main():
    plotter = plot_graphs_class()

if __name__ == "__main__":
    main()