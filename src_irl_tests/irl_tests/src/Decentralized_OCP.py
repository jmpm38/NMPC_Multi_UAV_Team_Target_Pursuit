from casadi import sin, cos, pi, tan
import casadi as ca
import numpy as np
import OCP2 as ocp
import sys_kin_2 as sys_kin
import yaml
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
        self.alpha_angle_ref = 0 #initial value for alpha

        #Iteration Counter
        self.iteration_one = True

    ###########################################################
    # Retrieve problem conditions from yaml file
    ###########################################################

    def yaml_handling(self):
        # Read the YAML file
        script_dir = os.path.dirname(__file__)

        # Construct the path to the file in directory `x`
        file_path = os.path.join(script_dir,'..','..', '..', 'config.yaml')

        # Open and load the YAML file
        with open(file_path, 'r') as yaml_file:
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

        # Build Cost Function and Set Constraints
        self.obj = 0
        self.standoff_dist_control_eff()
        self.kinematics_and_control_constraints()

        if(self.fov_terms):
            self.FOV_terms_n_constraints()
        
        #Define OCP cost function)
        self.opti.minimize(self.obj)

        #Set solver conditions
        self.solver_conditions()

    def initialize_ocp_variables_parameters(self):
        #Optimization Problem Parameters
        self.x_initial = self.opti.parameter(self.n_states, 1) # UAVs initial state for optimization
        self.target_pos = self.opti.parameter(3, self.N_horizon + 1) # Target position on the world frame throughout the horizon

        #Optimization Problem Variables - Controls and States
        self.U = self.opti.variable(self.n_controls, self.N_horizon) # Control Inputs
        self.X = self.opti.variable(self.n_states, self.N_horizon + 1) # States

        # Slack Variables
        self.S1 = self.opti.variable(self.N_horizon + 1) # Height Slack Variable
        if(self.fov_terms):
            self.S2 = self.opti.variable(self.N_horizon + 1) # Target inside camera FOV
            self.S3 = self.opti.variable(self.N_horizon + 1) # Target inside camera FOV

    ########################################################
    # Set UAV initial state and target position/velocity
    ########################################################

    def set_UAV_initial_state(self, x_initials):
        if(self.ocp_solved):
            if((np.max(self.sol.value(self.X[8, :]) - x_initials[8]))  > (2*np.pi)):
                valor = np.max(self.sol.value(self.X[8, :])) // (2*np.pi)
                x_initials[8] += valor*2*pi
            elif((np.min(self.sol.value(self.X[8, :]) - x_initials[8]))  < (-2*np.pi)):
                valor = np.min(self.sol.value(self.X[8, :])) // (-2*np.pi)
                x_initials[8] -= valor*2*pi
        self.opti.set_value(self.x_initial, x_initials)

    def set_target_position(self, target_pos, target_speed):
        self.opti.set_value(self.target_pos[:, 0], target_pos)
        aux0 = target_pos[0]
        aux1 = target_pos[1]
        aux2 = target_pos[2]
        for k in range(self.N_horizon):
            aux0 = aux0 + self.dt * target_speed[0]
            aux1 = aux1 + self.dt * target_speed[1]
            aux2 = aux2
            self.opti.set_value(self.target_pos[0, k+1], aux0)
            self.opti.set_value(self.target_pos[1, k+1], aux1)
            self.opti.set_value(self.target_pos[2, k+1], aux2)


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

            cos_horz_angle = ca.dot(horizontal, self.z_axis_camera)
            cos_vert_angle = ca.dot(vertical, self.z_axis_camera)

            horz_limit = cos(self.alpha_h/2)
            vert_limit = cos(self.alpha_v/2)

            self.opti.subject_to(cos_horz_angle + self.S2[k] > horz_limit)
            self.opti.subject_to(cos_vert_angle + self.S3[k] > vert_limit)

            # Add slack variable to cost function
            self.obj = self.obj + self.Q6 * (self.S2[k] ** 2) + self.Q6 * (self.S3[k] ** 2) 

            
    ###############################################################
    # UAV Kinematics and Control/States Limits
    ###############################################################

    def kinematics_and_control_constraints(self):
        # Initial state constraint
        self.opti.subject_to(self.X[:, 0] == self.x_initial)
        
        # Input and state bounds and system kinematics constraints
        for k in range(self.N_horizon+1):
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

                self.opti.subject_to(((self.U[3, k] - self.X[8,k]) ** 2) < 1)

            # State Variables Constraints
            self.opti.subject_to(self.X[0, k] > -500) 
            self.opti.subject_to(self.X[0, k] < 500) 
            self.opti.subject_to(self.X[1, k] > -500) 
            self.opti.subject_to(self.X[1, k] < 500) 
            self.opti.subject_to(self.X[2, k] + self.S1[k] > self.h_min) #UAV always h_min meters above the ground
            self.opti.subject_to(self.X[2, k] > 2) #also add hard constraint?
            # self.opti.subject_to(self.X[2, k] > self.h_min) #UAV always h_min meters above the ground
            self.opti.subject_to(self.X[2, k] < 50) 

            self.opti.subject_to(self.X[3, k] > -50) 
            self.opti.subject_to(self.X[3, k] < 50) 
            self.opti.subject_to(self.X[4, k] > -50) 
            self.opti.subject_to(self.X[4, k] < 50) 
            self.opti.subject_to(self.X[5, k] > -2.5) 
            self.opti.subject_to(self.X[5, k] < 2.5) 

            self.opti.subject_to(self.X[6, k] > -0.5) 
            self.opti.subject_to(self.X[6, k] < 0.5) 
            self.opti.subject_to(self.X[7, k] > -0.5) 
            self.opti.subject_to(self.X[7, k] < 0.5) 
            self.opti.subject_to(self.X[8, k] > (-50*np.pi)) 
            self.opti.subject_to(self.X[8, k] <= (50*np.pi)) 

            self.obj = self.obj + self.Q7*(self.S1[k] ** 2)

        
    ##########################################################
    # Set solver conditions
    ###########################################################

    def solver_conditions(self):
        # Set the solver options
        options = {
            'print_time': False,
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
            self.current_slack_alpha = np.array([0, 0, 0, 0])
            self.sol = self.opti.solve()
            self.ocp_solved = True
            self.warm_start()
            self.current_slack_alpha = np.array([self.sol.value(self.S1[0]), self.sol.value(self.S2[0]), self.sol.value(self.S3[0]), self.alpha_angle_ref])


            self.control_action = np.array([self.sol.value(self.X[5, 1]), self.sol.value(self.U[1, 0]),
                            self.sol.value(self.U[2, 0]), self.sol.value(self.U[3, 0])])
            self.iteration_one = False
            
        except RuntimeError as e:
            # Handle solver failure
            print("SOLVER ERROR", e)
            if(self.iteration_one):
                self.control_action = np.array([0.5, 0, 0, 0])
                self.current_slack_alpha = np.array([0, 0, 0, 0])


        return (self.control_action)
    
    def get_slack_alpha(self):
        return self.current_slack_alpha
        
    
def main():
    solving = decentralized_ocp_class()

if __name__ == "__main__":
    main() 
