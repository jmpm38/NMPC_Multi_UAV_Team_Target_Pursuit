#!/usr/bin/env python3

import numpy as np
import yaml
import rospy
from mrs_msgs.msg import UavState
from mrs_msgs.msg import HwApiAttitudeCmd
import math
import threading
import subprocess
import pickle
import Decentralized_OCP as dec_ocp
from std_msgs.msg import Float64MultiArray
import angular_optimization as ang_opt
import rospkg
import os

class one_opt:

    def __init__(self):

        # Receive variables from yaml file
        self.yaml_handling()

        # Create the Optimization Problem
        self.dec_ocp = dec_ocp.decentralized_ocp_class()
        self.ang_opt = ang_opt.angular_optimization(self.N_horizon, self.uav_num)

        # NMPC iteration counter
        self.iteration_counter = 0 
        self.target_position = np.zeros((3))
        self.target_speed = np.zeros((3))

        # Other UAVs predicted positions
        self.other_uavs = np.zeros((3, self.N_horizon + 1, 2))
        self.uavs_positions = np.zeros((3, self.N_horizon + 1, self.uav_num))

        # Initialize control commands
        self.calculate_throttle(9.81) # g = 9.81
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        # Initialize Simulation timer
        self.start_timer = 100000000000
        self.start_timer_2 = 100000000000
        self.recording_timer = 0

        # Auxiliary Flags
        self.not_cmd_kill = True
        self.msg_published = False
        self.start_periodic_task = False
        self.flag_target_info = False
        self.uav1_info = False
        self.uav2_info = False
        self.update_cost = False

        # Initialize arrays for saving uav states and inputs
        self.states_rec = np.zeros((9,1))
        self.controls_rec = np.zeros((4,1))

        # Initialize ROS
        rospy.init_node('uav1_publisher_subscriber', anonymous=True) # Anonymous makes it so that multiple listeners can be run simultaneously, ensuring different names
        self.pub = rospy.Publisher('/uav1/hw_api/attitude_cmd', HwApiAttitudeCmd, queue_size=10)
        self.pub_positions = rospy.Publisher('/uav1/predicted_positions', Float64MultiArray, queue_size=10)
        self.pub_slack_alpha = rospy.Publisher('/uav1/alpha_slack', Float64MultiArray, queue_size=10)
        # self.rate = rospy.Rate(10) #Hz
        self.data = None
        self.sub = rospy.Subscriber("/uav1/estimation_manager/uav_state", UavState, self.callback)
        self.sub_target = rospy.Subscriber("/uav1/target_estimate", Float64MultiArray, self.callback_target)
        self.sub_pos1 = rospy.Subscriber("/uav2/predicted_positions", Float64MultiArray, self.callback_positions1)
        self.sub_pos2 = rospy.Subscriber("/uav3/predicted_positions", Float64MultiArray, self.callback_positions2)

        # Create a thread for a periodic task
        self.periodic_thread = threading.Thread(target=self.periodic_task)
        self.periodic_thread.daemon = True  # Ensure thread exits when the main thread does
        
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

###################################################################################
# NMPC
###################################################################################

    def NMPC_logic(self):
        if((rospy.get_time() - self.start_timer) > self.sim_time):
            print("RECORDING TIME")
            self.data_recording()
        else:
            rospy.loginfo("Iteration: " + str(self.iteration_counter + 1))
        
            self.dec_ocp.set_UAV_initial_state(np.around(self.uav_state,4))
            self.dec_ocp.set_target_position(self.target_position, self.target_speed)
            # yaw_ref = np.around(np.arctan2((self.target_position[1] - self.uav_state[1]), (self.target_position[0] - self.uav_state[0])), 4)
            # self.dec_ocp.set_yaw_ref(yaw_ref)

            u_optimized = self.dec_ocp.solve()

            self.own_predicted_positions = self.dec_ocp.get_predicted_positions()

            self.iteration_counter = self.iteration_counter + 1

            if(rospy.get_time() - self.start_timer_2> 5):
                if(self.uav1_info and self.uav2_info):
                    if(not self.update_cost):
                        print("UPDATING REQS")
                        self.dec_ocp.decentralized_formation_requirements()
                        self.update_cost = True
                    # print("POSITION UPDATE")
                    # Angular Optimization
                    self.uavs_positions[:, :, 0] = self.dec_ocp.get_predicted_positions()
                    self.ang_opt.set_parameters(self.uavs_positions ,self.dec_ocp.get_target_positions())
                    alpha_angles = self.ang_opt.solve()
                    # self.dec_ocp.set_desired_angle(2.09)
                    self.dec_ocp.set_desired_angle(alpha_angles)
                    self.dec_ocp.set_UAVs_position(self.other_uavs)
                    self.uav1_info = False
                    self.uav2_info = False

            # Solver values to be published to the UAV
            self.calculate_throttle(u_optimized[0])
            self.roll = u_optimized[1]
            self.pitch = u_optimized[2]
            self.yaw = ((u_optimized[3] + np.pi) % (2*np.pi))-np.pi

            # Publish own predicted positions
            if((rospy.get_time() - self.start_timer) > 0.5):
                self.publish_positions()
                self.start_timer = rospy.get_time()

            self.publish_alpha_slack()
                
             
###################################################################################
# ROS
###################################################################################

    def callback(self, data):
        self.data = data

        # Data processing
        orientation = self.euler_from_quaternion(data.pose.orientation.x, data.pose.orientation.y,
                                                data.pose.orientation.z, data.pose.orientation.w)
        
        # Set UAV current state
        uav_state_aux = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z,
                          data.velocity.linear.x, data.velocity.linear.y, data.velocity.linear.z,
                          orientation[0], orientation[1], orientation[2]])
        
        self.uav_state = np.transpose(uav_state_aux)

        # if((self.iteration_counter > 0)): #and ((rospy.get_time() - self.recording_timer) > 0.08)):
        #     if(self.iteration_counter == 0):
        #         self.states_rec[:, 0] = uav_state_aux
        #         self.controls_rec[:, 0] = self.sol.value(self.U[:, 0])
        #     else:
        #         self.states_rec = np.column_stack((self.states_rec, uav_state_aux))
        #         self.controls_rec = np.column_stack((self.controls_rec, self.sol.value(self.U[:, 0])))
        #     self.recording_timer = rospy.get_time()
        # print(orientation[2])
        if(self.iteration_counter>1):
            self.publisher()

    def callback_target(self, data):
        self.target_position[0] = data.data[0]
        self.target_position[1] = data.data[1]
        self.target_speed[0] = data.data[2]
        self.target_speed[1] = data.data[3]
        self.flag_target_info = True

    def publisher(self):        

        # Build command message
        cmd_msg = HwApiAttitudeCmd()

        #self.yaw = np.arctan2((self.target_position[1] - self.uav_state[1]), (self.target_position[0] - self.uav_state[0]))
        # Convert Euler Angles to Quaternions
        quaternion = self.get_quaternion_from_euler(self.roll, self.pitch, self.yaw)

        cmd_msg.orientation.x = quaternion[0]
        cmd_msg.orientation.y = quaternion[1]
        cmd_msg.orientation.z = quaternion[2]
        cmd_msg.orientation.w = quaternion[3]

        cmd_msg.throttle = self.throttle

        self.pub.publish(cmd_msg)
        self.msg_published = True
        #rospy.loginfo("PUB: " + str(self.throttle) + " " + str(self.roll) + " " + str(self.pitch) + " " + str(self.yaw))

    def run(self):
        # Spin to keep the script for exiting
        while not rospy.is_shutdown():
            if (self.data):
                if((not self.start_periodic_task) and (self.flag_target_info)):
                # if(self.msg_published and (not self.start_periodic_task) and (self.flag_target_info)):
                    # Start the periodic task thread
                    self.periodic_thread.start()
                    self.start_periodic_task = True

                if((self.iteration_counter > 3) and self.not_cmd_kill):
                    #Shutting down Control Manager
                    command = "rosnode kill /uav1/control_manager"
                    try:
                        subprocess.run(command, shell=True, check=True)
                    except:
                        pass
                    print("SHUT DOWN: Control_Manager")
                    self.not_cmd_kill = False
                    self.start_timer = rospy.get_time()
                    self.start_timer_2 = rospy.get_time()
                  
    def periodic_task(self):
        rate = rospy.Rate(5)  # 5 Hz

        # Perform periodic task
        while not rospy.is_shutdown():
            self.NMPC_logic()
            rate.sleep()

    def publish_positions(self):
        aux1 = self.own_predicted_positions.flatten()
        msg_positions = Float64MultiArray()
        msg_positions.data = aux1
        self.pub_positions.publish(msg_positions)
        
    def callback_positions1(self, data):
        aux1 = np.array(data.data)
        aux1 = aux1.reshape(3, self.N_horizon + 1)
        self.other_uavs[:, :, 0] = aux1
        self.uavs_positions[:, :, 1] = aux1
        self.uav1_info = True

    def callback_positions2(self,data):
        aux1 = np.array(data.data)
        aux1 = aux1.reshape(3, self.N_horizon + 1)
        self.other_uavs[:, :, 1] = aux1
        self.uavs_positions[:, :, 2] = aux1
        self.uav2_info = True

    def publish_alpha_slack(self):
        aux1 = self.dec_ocp.get_slack_alpha()
        msg_slack_alpha = Float64MultiArray()
        msg_slack_alpha.data = aux1
        self.pub_slack_alpha.publish(msg_slack_alpha)
        
###################################################################################
# AUXILIAR FUNCTIONS
###################################################################################

    def euler_from_quaternion(self, x, y, z, w):
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
    
    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.
        
        Input
            :param roll: The roll (rotation around x-axis) angle in radians.
            :param pitch: The pitch (rotation around y-axis) angle in radians.
            :param yaw: The yaw (rotation around z-axis) angle in radians.
        
        Output
            :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return [qx, qy, qz, qw]
    
    def calculate_throttle(self, thrust_acc):
        # mass= 2 #kg
        # nr_motors = 4
        # force_per_uav = mass * thrust_acc / nr_motors
        # a = 0.28980
        # b = -0.17647
        mass= 2.3 #kg
        nr_motors = 6
        force_per_uav = mass * thrust_acc / nr_motors
        a = 0.37119
        b = -0.17647

        self.throttle = a * np.sqrt(force_per_uav) + b
        
        if(self.throttle < 0):
            self.throttle = 0.0
        if(self.throttle > 1):
            self.throttle = 1.0

    def data_recording(self):
        self.N_horizon = self.states_rec.shape[1]
        states = np.column_stack((self.states_rec, self.states_rec[:, -1]))
        controls = self.controls_rec
        target_positions = np.array([
        [self.target_pos[0]] * (self.N_horizon + 1),
        [self.target_pos[1]] * (self.N_horizon + 1),
        [self.target_pos[2]] * (self.N_horizon + 1)])

        # Saving the objects:
        with open('true_sim.pkl', 'wb') as f_2:  # Python 3: open(..., 'wb')
            print("SAVING VARIABLES")
            pickle.dump([target_positions, states, controls, self.N_horizon], f_2)
        
##################################################################################

def main():
    solving = one_opt()
    solving.run()

if __name__ == "__main__":
    main()   




#### COMMANDS
# need to add solving time to alpha_slack
# rosbag record -O simulation_1 /uav1/estimation_manager/uav_state /uav1/hw_api/attitude_cmd /uav1/target_estimate /uav1/alpha_slack 
# rosbag record -O simulation_2 /uav2/estimation_manager/uav_state /uav2/hw_api/attitude_cmd /uav1/target_estimate /uav2/alpha_slack
# rosbag record -O simulation_3 /uav3/estimation_manager/uav_state /uav3/hw_api/attitude_cmd /uav1/target_estimate /uav3/alpha_slack
# source devel/setup.bash

# NEEDED CHANGES:
# control_manager shut down
# subscribers and publishers
# node name