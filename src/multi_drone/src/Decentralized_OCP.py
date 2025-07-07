from casadi import sin, cos, pi, tan
import casadi as ca
import numpy as np
import OCP2 as ocp
import sys_kin_2 as sys_kin
import yaml
import rospy
import target_graphs2 as plotter
import rospkg
import os

class decentralized_ocp_class:

    def __init__(self):
        # Receive variables from yaml file
        self.yaml_handling()

        #Auxiliary Variables
        self.ocp_solved = False

        # Call to the System Kinematics Class
        self.kin_class = sys_kin.UAV_kin_class(self.dt)
        self.kinematics = self.kin_class.uav_kinematics()
        self.n_states = self.kin_class.number_states()
        self.n_controls = self.kin_class.number_controls()

        # Create the Optimization Problem
        self.create_ocp()

        #Auxiliary Variables
        # self.slack_variables_alpha_value = np.zeros((4,1)) #array to store slack variables value
        self.alpha_angle_ref = 0 #initial value for alpha
        self.iter_aux = 0

        self.max_steps = int(self.sim_time / self.dt)*0.75


    ###########################################################
    # Retrieve problem conditions from yaml file
    ###########################################################

    def yaml_handling(self):
        # Initialize ROS package manager
        rospack = rospkg.RosPack()

        # Get the path of your ROS package
        package_path = rospack.get_path('multi_drone')  # Replace 'my_package' with your package name

        # Construct the full path to the YAML file in the config folder
        yaml_file_path = os.path.join(package_path, 'config', 'config.yaml')
        # Read the YAML file
        with open(yaml_file_path, 'r') as yaml_file:
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
        self.Q7 = Weights["Q7"]
        self.Q8 = Weights["Q8"]
        
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

    ########################################################
    # Create OCP
    #######################################################

    def create_ocp(self):
        # Initialize Solver
        self.opti = ca.Opti()

        
        # Initialize variables and parameters
        self.initialize_ocp_variables_parameters()

        if((self.uav_num > 1) and (self.formation_terms)):
            self.set_UAV_decentralized_positions_param()

        # Build Cost Function and Set Constraints
        self.obj = 0
        self.standoff_dist_control_eff()
        self.kinematics_and_control_constraints()

        if(self.fov_terms):
            self.FOV_terms_n_constraints()
        
        # self.energy_efficiency()

        #Define OCP cost function)
        self.opti.minimize(self.obj)

        #Set solver conditions
        self.solver_conditions()

    def initialize_ocp_variables_parameters(self):
        #Optimization Problem Parameters
        self.x_initial = self.opti.parameter(self.n_states, 1) # UAVs initial state for optimization
        self.target_pos = self.opti.parameter(3, self.N_horizon + 1) # Target position on the world frame throughout the horizon
        self.yaw_ref = self.opti.parameter()

        #Optimization Problem Variables - Controls and States
        self.U = self.opti.variable(self.n_controls, self.N_horizon) # Control Inputs
        self.X = self.opti.variable(self.n_states, self.N_horizon + 1) # States

        # Slack Variables
        self.S1 = self.opti.variable(self.N_horizon + 1) # Height Slack Variable
        if(self.fov_terms):
            self.S2 = self.opti.variable(self.N_horizon + 1) # Target inside camera FOV
            self.S3 = self.opti.variable(self.N_horizon + 1) # Target inside camera FOV
        self.lambda_slack_1 = self.opti.variable(self.N_horizon + 1) # obstacle slack Variable
        self.lambda_slack_2 = self.opti.variable(self.N_horizon + 1) # obstacle slack Variable

    ########################################################
    # Set UAV initial state and target position/velocity
    ########################################################

    def set_UAV_initial_state(self, x_initials):
        # print("actual state: " + str(x_initials[8]))
        if(self.ocp_solved):
            if((np.max(self.sol.value(self.X[8, :]) - x_initials[8]))  > (2*np.pi)):
                valor = np.max(self.sol.value(self.X[8, :])) // (2*np.pi)
                x_initials[8] += valor*2*pi#getting yaw value
            elif((np.min(self.sol.value(self.X[8, :]) - x_initials[8]))  < (-2*np.pi)):
                valor = np.min(self.sol.value(self.X[8, :])) // (-2*np.pi)
                x_initials[8] -= valor*2*pi#getting yaw value

        # print("controller state: " + str(x_initials[8]))
        # print("UAV STATE: " + str(x_initials))
        self.opti.set_value(self.x_initial, x_initials)

    def set_target_position(self, target_pos, target_speed):
        # print("TARGET INFO: " + str(target_pos) + " s:   " + str(target_speed))
        # Target Position with constant velocity target model
        self.np_target = np.zeros((3, self.N_horizon + 1))
        self.opti.set_value(self.target_pos[:, 0], target_pos)
        aux0 = target_pos[0]
        aux1 = target_pos[1]
        aux2 = target_pos[2]
        self.np_target[:, 0] = target_pos
        for k in range(self.N_horizon):
            aux0 = aux0 + self.dt * target_speed[0]
            aux1 = aux1 + self.dt * target_speed[1]
            aux2 = aux2
            self.np_target[0, k+1] = aux0
            self.np_target[1, k+1] = aux1
            self.np_target[2, k+1] = aux2
            self.opti.set_value(self.target_pos[0, k+1], aux0)
            self.opti.set_value(self.target_pos[1, k+1], aux1)
            self.opti.set_value(self.target_pos[2, k+1], aux2)

    def set_yaw_ref(self, yaw_ref_value):
        # self.opti.set_value(self.yaw_ref, yaw_ref_value)
        self.opti.set_value(self.yaw_ref, yaw_ref_value)

    ###############################################################
    # Target Tracking Cost Function and Constraints
    ###############################################################

    def standoff_dist_control_eff(self, ):
        for k in range(self.N_horizon + 1):

            # Distance to target term
            current_uav_pos = ca.vertcat(self.X[0, k], self.X[1, k], self.X[2, k])
            dist_term = ocp.dist_term_ocp(current_uav_pos, self.target_pos[:, k], self.Q1, self.Standoff_distance)

            # Control action term
            if(k != self.N_horizon):
                current_control_term = ca.vertcat(self.U[0, k], self.U[1, k], self.U[2, k], self.U[3, k])
                body_frame = ocp.calculate_body_frame_coords(self.target_pos[:, k], current_uav_pos, self.X[6, k], self.X[7, k], self.X[8, k])
                epsilon = 1e-6
                # yaw_ref = ca.atan2(body_frame[1] + epsilon, body_frame[0] + epsilon)
                yaw_ref = ca.atan2(self.target_pos[1, k] - self.X[1, k] + epsilon, self.target_pos[0, k] - self.X[0, k] + epsilon)
                if(self.ocp_solved):
                    if((np.max(self.sol.value(self.X[8, :]) - yaw_ref))  > (2*np.pi)):
                        valor = np.max(self.sol.value(self.X[8, :])) // (2*np.pi)
                        yaw_ref += valor*2*pi#getting yaw value
                    elif((np.min(self.sol.value(self.X[8, :]) - yaw_ref))  < (-2*np.pi)):
                        valor = np.min(self.sol.value(self.X[8, :])) // (-2*np.pi)
                        yaw_ref -= valor*2*pi#getting yaw value
                control_term = ocp.control_term_ocp (current_control_term, self.U_ref, self.Q2, yaw_ref)
            
            # RPY rate term
            current_rpy = ca.vertcat(self.X[6, k], self.X[7, k], self.X[8, k])#, self.X[8, k])
            if(k > 0):
                rpy_rate = ca.minus(current_rpy, past_rpy) / self.dt
                rpy_rate_term = self.Q8 * ca.sumsqr(rpy_rate)
                self.obj = self.obj + rpy_rate_term
            past_rpy = current_rpy

            # Add terms to cost function
            self.obj = self.obj + dist_term
            self.obj = self.obj + control_term

    def FOV_terms_n_constraints(self, ):
        epsilon = 1e-6
        for k in range(self.N_horizon + 1):

            # Centered FOV term
            current_uav_pos = ca.vertcat(self.X[0, k], self.X[1, k], self.X[2, k])
            # Calculate relative position in camera frame
            rel_pos = ocp.calculate_beta_ocp(self.target_pos[:, k], current_uav_pos, self.X[6, k], self.X[7, k], self.X[8, k], self.z_rotation_angle)
            rel_pos_norm = ca.norm_2(rel_pos)
            rel_pos_versor = rel_pos*(1/rel_pos_norm)
            beta_angle = ca.dot(rel_pos_versor, self.z_axis_camera)
            fov_term = ocp.centered_FOV_ocp(beta_angle, self.Q3)

            # Add to Cost Function
            self.obj = self.obj + fov_term

            # Constraints:
            horizontal_vec = ca.vertcat(rel_pos_versor[0], 0, rel_pos_versor[2])
            vertical_vec = ca.vertcat(0, rel_pos_versor[1], rel_pos_versor[2])

            horizontal_norm = ca.norm_2(horizontal_vec)
            vertical_norm = ca.norm_2(vertical_vec)

            horizontal = horizontal_vec * (1/horizontal_norm)
            vertical = vertical_vec * (1/vertical_norm)

            cos_horz_angle = ca.dot(horizontal, self.z_axis_camera) #* (rel_pos_versor[0]/rel_pos_versor[0])
            cos_vert_angle = ca.dot(vertical, self.z_axis_camera) #* (rel_pos_versor[1]/rel_pos_versor[1])

            # Beta angle derivative term
            if(k>0):
                fov_dot_term = ocp.centered_FOV_dot_ocp(beta_angle, beta_past, self.dt, self.Q4)
                # fov_dot_term_2 = ocp.centered_FOV_dot_ocp(beta_angle, beta_past, self.dt, self.Q4)
                # fov_dot_term_3 = ocp.centered_FOV_dot_ocp(beta_angle, beta_past, self.dt, self.Q4)
            #     self.obj = self.obj + fov_dot_term

            beta_past = beta_angle
            # beta_past_2 = cos_horz_angle
            # beta_past_3 = cos_vert_angle

            # fov_term_2 = ocp.centered_FOV_ocp(cos_horz_angle, self.Q3)
            # fov_term_3 = ocp.centered_FOV_ocp(cos_vert_angle, self.Q3)
            # self.obj = self.obj + fov_term_2 + fov_term_3

            horz_limit = cos(self.alpha_h/2)
            vert_limit = cos(self.alpha_v/2)

            self.opti.subject_to(cos_horz_angle + self.S2[k] > horz_limit)
            self.opti.subject_to(cos_vert_angle + self.S3[k] > vert_limit)
            # Add slack variable to cost function
            self.obj = self.obj + self.Q6 * (self.S2[k] ** 2) + self.Q6 * (self.S3[k] ** 2) 


            

    def energy_efficiency(self):
        for k in range(self.N_horizon):
            current_control_term = ca.vertcat(self.U[0, k], self.U[1, k], self.U[2, k], self.U[3, k])
            epsilon = 1e-6
            yaw_ref = ca.atan2(self.target_pos[1, k] - self.X[1, k] + epsilon, self.target_pos[0, k] - self.X[0, k] + epsilon)
            U_ref = ca.vertcat(9.81, 0, 0, yaw_ref)
            deviation = ca.minus(current_control_term, U_ref)
            self.obj = self.obj + 1000 * (ca.fmax(0, deviation[0]) ** 2)
            # self.obj = self.obj + 10000 * ((deviation[1] ** 2) + (deviation[2] ** 2)) #+ (deviation[3] ** 2))

    ###############################################################
    # UAV Kinematics and Control/States Limits
    ###############################################################

    def kinematics_and_control_constraints(self):
        # Initial state constraint
        self.opti.subject_to(self.X[:, 0] == self.x_initial)
        
        # Input and state bounds and system kinematics constraints
        for k in range(self.N_horizon+1):
            # if(k < self.N_horizon-1):
            #     self.opti.subject_to(((self.U[0, k+1] - self.U[0, k]) ** 2) < 4)
            if(k != self.N_horizon):
                # UAV Kinematics
                self.opti.subject_to(self.X[:, k+1] == self.kinematics(self.X[:, k], self.U[:, k]))

                # Control Input Constraints
                self.opti.subject_to(self.U[0, k] < self.T_max)
                self.opti.subject_to(self.T_min < self.U[0, k])
                self.opti.subject_to(self.U[1, k] < self.pitch_max)
                self.opti.subject_to(-self.pitch_max < self.U[1, k])
                self.opti.subject_to(self.U[2, k] < self.roll_max)
                self.opti.subject_to(-self.roll_max < self.U[2, k])
                self.opti.subject_to(self.U[3, k] < 50*np.pi)
                self.opti.subject_to((-50*np.pi) < self.U[3, k])

                self.opti.subject_to(((self.U[3, k] - self.X[8,k]) ** 2) < 0.5)

            # # State Variables Constraints
            # self.opti.subject_to(self.X[0, k] > -75) 
            # self.opti.subject_to(self.X[0, k] < 75) 
            # self.opti.subject_to(self.X[1, k] > -75) 
            # self.opti.subject_to(self.X[1, k] < 75) 
            # self.opti.subject_to(self.X[2, k] + self.S1[k] > self.h_min) #UAV always h_min meters above the ground
            # self.opti.subject_to(self.X[2, k] > 2) #also add hard constraint?
            # # self.opti.subject_to(self.X[2, k] > self.h_min) #UAV always h_min meters above the ground
            # self.opti.subject_to(self.X[2, k] < 50) 

            # self.opti.subject_to(self.X[3, k] > -10) 
            # self.opti.subject_to(self.X[3, k] < 10) 
            # self.opti.subject_to(self.X[4, k] > -10) 
            # self.opti.subject_to(self.X[4, k] < 10) 
            # self.opti.subject_to(self.X[5, k] > -10) 
            # self.opti.subject_to(self.X[5, k] < 10) 

            # self.opti.subject_to(self.X[6, k] > -0.4) 
            # self.opti.subject_to(self.X[6, k] < 0.4) 
            # self.opti.subject_to(self.X[7, k] > -0.4) 
            # self.opti.subject_to(self.X[7, k] < 0.4) 
            # self.opti.subject_to(self.X[8, k] > (-50*np.pi)) 
            # self.opti.subject_to(self.X[8, k] <= (50*np.pi)) 

            # State Variables Constraints
            self.opti.subject_to(self.X[0, k] > -1000) 
            self.opti.subject_to(self.X[0, k] < 1000) 
            self.opti.subject_to(self.X[1, k] > -1000) 
            self.opti.subject_to(self.X[1, k] < 1000) 
            self.opti.subject_to(self.X[2, k] + self.S1[k] > self.h_min) #UAV always h_min meters above the ground
            self.opti.subject_to(self.X[2, k] > 2) #also add hard constraint?
            # self.opti.subject_to(self.X[2, k] > self.h_min) #UAV always h_min meters above the ground
            self.opti.subject_to(self.X[2, k] < 50) 

            self.opti.subject_to(self.X[3, k] > -30) 
            self.opti.subject_to(self.X[3, k] < 30) 
            self.opti.subject_to(self.X[4, k] > -30) 
            self.opti.subject_to(self.X[4, k] < 30) 
            self.opti.subject_to(self.X[5, k] > -30) 
            self.opti.subject_to(self.X[5, k] < 30) 

            self.opti.subject_to(self.X[6, k] > -0.7) 
            self.opti.subject_to(self.X[6, k] < 0.7) 
            self.opti.subject_to(self.X[7, k] > -0.7) 
            self.opti.subject_to(self.X[7, k] < 0.7) 
            self.opti.subject_to(self.X[8, k] > (-50*np.pi)) 
            self.opti.subject_to(self.X[8, k] <= (50*np.pi)) 

            self.obj = self.obj + self.Q7*(self.S1[k] ** 2)

            # Yaw Smooth Constraint
            # if((k > 0) and (k != self.N_horizon)):
            #     self.opti.subject_to(ca.fabs(self.U[3, k] - self.U[3, k-1]) <= 0.2)
    
    ###############################################################
    # Formation Requirements Cost Function and Constraints
    ###############################################################
    
    # Set other UAVs position parameters and update
    def set_UAV_decentralized_positions_param(self):
        self.uavs = []
        for i in range(self.uav_num - 1):
            self.uavs.append(self.opti.parameter(3, self.N_horizon + 1))
        self.desired_angle = self.opti.parameter(self.N_horizon + 1)

    def set_UAVs_position(self, uavs_vec):
        # print("UAVS VEC")
        # print(uavs_vec)
        for i in range(self.uav_num - 1):
            self.opti.set_value(self.uavs[i][:,:], uavs_vec[:, :, i]) 
    
    def set_desired_angle(self, desired_angle):
        self.alpha_angle_ref = desired_angle
        self.opti.set_value(self.desired_angle, desired_angle)

    # Formation Requirements Terms and Constraints
    def decentralized_formation_requirements(self):
        #Safety Distance Soft Constrsaint
        for k in range(self.N_horizon + 1):
            for uav_i in range((self.uav_num - 1)):
                current_uav_pos = ca.vertcat(self.X[0, k], self.X[1, k], self.X[2, k])
                other_uav = self.uavs[uav_i][:, k]
                dist_vector = ca.minus(current_uav_pos, other_uav)
                distance = ca.sumsqr(dist_vector) #squared distance

            #TODO:
            # Set soft constraint
            # self.opti.subject_to(distance + self.S_4[0, k] > (self.safety_distance ** 2))
            # Add slack variable to cost function
            # self.obj = self.obj + self.Q7 * (self.S_4[0, k] ** 2)

        #Fisher Information Matrix
        for k in range(self.N_horizon + 1):
            current_uav_pos = ca.vertcat(self.X[0, k], self.X[1, k], self.X[2, k])
    
            # Vector from UAV to target
            target_vector = self.target_pos[:, k] - current_uav_pos
            
            # UAV heading vector based on the desired angle (assumed to be 2D for simplicity)
            uav_heading = ca.vertcat(ca.cos(self.desired_angle[k]), ca.sin(self.desired_angle[k]))
            
            # Calculate cosine of the angle error using the dot product
            cos_alpha_error = (target_vector[0] * uav_heading[0] + target_vector[1] * uav_heading[1]) / (ca.norm_2(target_vector) * ca.norm_2(uav_heading))
            
            # Ensure the value is between -1 and 1 for numerical stability
            alpha_error = ca.acos(ca.fmax(cos_alpha_error, -1.0))
            
            # Update objective function with the squared angle error
            self.obj = self.obj + self.Q5 * (alpha_error ** 2)

            # for uav_i in range((self.uav_num)):
            #     # For each UAV we need to calculate alpha: uav angle relative to the target and the distance to the target
            #     if(uav_i == 0):
            #         # Calculate Alpha current UAV
            #         current_uav_pos = ca.vertcat(self.X[0, k], self.X[1, k], self.X[2, k])
            #         alpha = ca.atan2(self.target_pos[1, k] - current_uav_pos[1], self.target_pos[0, k] - current_uav_pos[0])
            #         alphas_vec = alpha

            #         # Calculate distance to target current UAV
            #         dist_vector = ca.minus(current_uav_pos, self.target_pos[:, k])
            #         distance_sqr = ca.sumsqr(dist_vector) #squared distance
            #         distance_sqr_vec = distance_sqr

            #     else:
            #         # Calculate Alpha other UAVs
            #         current_uav_pos = self.uavs[uav_i-1][:, k]
            #         alpha = ca.atan2(self.target_pos[1, k] - current_uav_pos[1], self.target_pos[0, k] - current_uav_pos[0])
            #         alphas_vec = ca.vertcat(alphas_vec, alpha)

            #         # Calculate distance to target other UAVs
            #         dist_vector = ca.minus(current_uav_pos, self.target_pos[:, k])
            #         distance_sqr = ca.sumsqr(dist_vector) #squared distance
            #         distance_sqr_vec = ca.vertcat(distance_sqr_vec, distance_sqr)

            # for i in range(1, self.uav_num, 1):
            #     alpha_1 = alphas_vec[0]
            #     alpha_2 = alphas_vec[i]
            #     denominator = ca.sin(alpha_1 - alpha_2) ** 2

            #     dist_1 = distance_sqr_vec[0]
            #     dist_2 = distance_sqr_vec[i]
            #     numerator = dist_1 * dist_2
            #     if (i == 1):
            #         fim_value = numerator#/denominator
            #     else:
            #         fim_value = fim_value + numerator#(numerator/denominator)
            # self.obj = self.obj + self.Q5 * (fim_value ** 2)
        
        self.opti.minimize(self.obj)
        
    ##########################################################
    # Set solver conditions
    ###########################################################

    def solver_conditions(self):
        # Set the solver options
        options = {
            'print_time': True,
            'expand': True,  # Expand makes function evaluations faster but requires more memory
            'ipopt': {
                'print_level': 0,
                'tol': 5e-1,
                'dual_inf_tol': 5.0,
                'constr_viol_tol': 1e-1,
                'compl_inf_tol': 1e-1,
                'acceptable_tol': 1e-2,
                'acceptable_constr_viol_tol': 0.01,
                'acceptable_dual_inf_tol': 1e10,
                'acceptable_compl_inf_tol': 0.01,
                'acceptable_obj_change_tol': 1e20,
                'diverging_iterates_tol': 1e20,
                'warm_start_bound_push': 1e-4,
                'warm_start_bound_frac': 1e-4,
                'warm_start_slack_bound_frac': 1e-4,
                'warm_start_slack_bound_push': 1e-4,
                'warm_start_mult_bound_push': 1e-4,
            },
            'verbose': False,
        }
        
        # Set the solver with the options
        self.opti.solver('ipopt', options)

    def warm_start(self):
        # rospy.loginfo("PREDICTED STATE: " + str(self.sol.value(self.X[8, :])))
        self.opti.set_initial(self.X[:, 1:self.N_horizon],self.sol.value(self.X[:, 1:self.N_horizon]))
        self.opti.set_initial(self.U[:, 1:(self.N_horizon-1)],self.sol.value(self.U[:, 1:(self.N_horizon-1)]))

        self.opti.set_initial(self.S1[1:(self.N_horizon-1)], self.sol.value(self.S1[1:(self.N_horizon-1)]))
        if(self.fov_terms):
            self.opti.set_initial(self.S2[1:(self.N_horizon-1)], self.sol.value(self.S2[1:(self.N_horizon-1)]))
            self.opti.set_initial(self.S3[1:(self.N_horizon-1)], self.sol.value(self.S3[1:(self.N_horizon-1)]))

    def solve(self):
        try:
            self.sol = self.opti.solve()
            self.ocp_solved = True
        except RuntimeError as e:
            # Handle solver failure
            print("Solver failed with error:", e)
            print("Intermediate state X:", self.opti.debug.value(self.X[0, :]))
            print("Intermediate state X:", self.opti.debug.value(self.X[1, :]))
            print("Intermediate state X:", self.opti.debug.value(self.X[8, :]))
            arr = np.array([[20, 20, 0]]).T  # Column vector
            arr = np.tile(arr, (1, self.N_horizon + 1))

            # plt = plotter.plot_graphs_class()
            # plt.set_variables(arr, self.opti.debug.value(self.X[:, :]), self.opti.debug.value(self.U[:, :]), self.N_horizon, np.arange(0, self.N_horizon + 1), np.arange(0, self.N_horizon ), np.arange(0, self.N_horizon + 1))
            # plt.plot_starter()
            # input()
        # self.opt_time = 0.2
        self.opt_time = self.sol.stats()['t_wall_total']
        # self.optimization_times(opt_time)
        self.warm_start()
        self.create_slack_variable_array_alpha()

        control_action = np.array([self.sol.value(self.U[0, 0]), self.sol.value(self.U[1, 0]),
                          self.sol.value(self.U[2, 0]), self.sol.value(self.U[3, 0])])


        

        return (control_action)

    def get_predicted_positions(self):
        positions_over_horizon = np.zeros([3, self.N_horizon + 1])
        positions_over_horizon[0, :] = self.sol.value(self.X[0, :])
        positions_over_horizon[1, :] = self.sol.value(self.X[1, :])
        positions_over_horizon[2, :] = self.sol.value(self.X[2, :])
        # print(positions_over_horizon)
        # print("TARGET: ")
        # print(self.np_target)
        # input()
        return positions_over_horizon # 3, N_horizon + 1

    def get_target_positions(self):
        return self.np_target
    
    def create_slack_variable_array_alpha(self):
        self.current_slack_alpha = np.array([self.sol.value(self.S1[0]), self.sol.value(self.S2[0]), self.sol.value(self.S3[0]), self.alpha_angle_ref, self.opt_time])
        # self.slack_variables_alpha_value = np.hstack((self.slack_variables_alpha_value, self.current_slack))

    def get_slack_alpha(self):
        return self.current_slack_alpha
    
    # def optimization_times(self, opt_time):
    #     if(self.iter_aux == 0):
    #         self.opt_times = np.array([opt_time])
    #     else:
    #         self.opt_times = np.append(self.opt_times, np.array([opt_time]))
    #     self.iter_aux = self.iter_aux + 1
    #     if(self.iter_aux >= self.max_steps):
    #         self.retrieve_times()

    # def retrieve_times(self):
    #     print("FIRST OPTIMIZATION TIME: ")
    #     print(self.opt_times[0])
    #     print("AVERAGE OPTIMIZATION TIME:")
    #     print(np.average(self.opt_times[1:self.opt_times.size]))
    #     print("MAX OPTIMIZATION TIME:")
    #     print(np.max(self.opt_times[1:self.opt_times.size]))
    #     print("MIN OPTIMIZATION TIME:")
    #     print(np.min(self.opt_times[1:self.opt_times.size]))
    #     input()
        
    
def main():
    solving = decentralized_ocp_class()

if __name__ == "__main__":
    main() 