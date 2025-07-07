#!/usr/bin/env python3

import numpy as np
import yaml
import rospy
import math
import threading
import Decentralized_OCP as dec_ocp
from std_msgs.msg import Float64MultiArray
from mavros_msgs.msg import AttitudeTarget
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
import os


class one_opt:

    def __init__(self):

        # Receive variables from yaml file
        self.yaml_handling()

        # Create the Optimization Problem
        self.dec_ocp = dec_ocp.decentralized_ocp_class()

        self.target_position = np.zeros((3))
        self.target_speed = np.zeros((3))

        # Initialize control commands
        self.throttle = 0.5
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.slack_data = [0.0, 0.0, 0.0, 0.0]

        # Auxiliary Flags
        self.msg_published = False
        self.start_periodic_task = False
        self.flag_target_info = False
        self.velocity_body = [0, 0, 0]

        # Initialize ROS
        rospy.init_node('uav_publisher_subscriber', anonymous=True)
        self.pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
        self.pub_slack_alpha = rospy.Publisher('/uav1/alpha_slack', Float64MultiArray, queue_size=10)

        self.data = None
        self.sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.callback)
        self.sub_2 = rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, self.callback_2)
        self.sub_target = rospy.Subscriber("/uav1/target_estimate", Float64MultiArray, self.callback_target)

        # Create a thread for a periodic task
        self.periodic_thread = threading.Thread(target=self.NMPC_logic)
        self.callback_thread = threading.Thread(target=self.callback_spin)
        self.publisher_thread = threading.Thread(target=self.publisher)
        self.periodic_thread.daemon = True  # Ensure thread exits when the main thread does

        self.callback_thread.start()
        self.publisher_thread.start()
        
    def yaml_handling(self):
        # Read the YAML file
        # Get the current file's directory
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
        rate = rospy.Rate(6)  # 5 Hz

        while not rospy.is_shutdown():
            rospy.loginfo("NMPC")

            self.dec_ocp.set_UAV_initial_state(np.around(self.uav_state, 6))
            self.dec_ocp.set_target_position(self.target_position, self.target_speed)

            u_optimized = self.dec_ocp.solve()

            self.throttle = u_optimized[0]
            self.calculate_climb_rate()
            self.roll = u_optimized[1]
            self.pitch = u_optimized[2]
            self.yaw = ((u_optimized[3] + np.pi) % (2*np.pi))-np.pi

            self.slack_data = self.dec_ocp.get_slack_alpha()

            rate.sleep()
             
###################################################################################
# ROS
###################################################################################

    def callback(self, data):
        self.data = data

        # Data processing
        orientation = self.euler_from_quaternion(data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                                                data.pose.pose.orientation.z, data.pose.pose.orientation.w)
        
        # Set UAV current state
        uav_state_aux = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z,
                          self.velocity_body[0], self.velocity_body[1], self.velocity_body[2],
                          orientation[0], orientation[1], orientation[2]])
        
        self.uav_state = np.transpose(uav_state_aux)

    def callback_2(self, data):
        self.velocity_body = np.array([data.twist.linear.x, data.twist.linear.y, data.twist.linear.z])
        
    def callback_target(self, data):
        self.target_position[0] = data.data[0]
        self.target_position[1] = data.data[1]
        self.target_speed[0] = data.data[2]
        self.target_speed[1] = data.data[3]
        self.flag_target_info = True

    def callback_spin(self):
        # Handles callbacks on a separate thread/core
        rospy.spin()

    def publisher(self):    
        rate = rospy.Rate(10)  # Publishing rate in Hz
        while not rospy.is_shutdown():    
            cmd_msg = AttitudeTarget()

            quaternion = self.get_quaternion_from_euler(self.roll, self.pitch, self.yaw)

            cmd_msg.orientation.x = quaternion[0]
            cmd_msg.orientation.y = quaternion[1]
            cmd_msg.orientation.z = quaternion[2]
            cmd_msg.orientation.w = quaternion[3]
            
            cmd_msg.thrust = self.throttle

            self.pub.publish(cmd_msg)
            self.msg_published = True

            self.publish_alpha_slack()

            rate.sleep()

    def publish_alpha_slack(self):
        msg_slack_alpha = Float64MultiArray()
        msg_slack_alpha.data = self.slack_data
        self.pub_slack_alpha.publish(msg_slack_alpha)

    def run(self):
        # Spin to keep the script for exiting
        while not rospy.is_shutdown():
            if (self.data):
                if(self.msg_published and (not self.start_periodic_task) and (self.flag_target_info)):
                    # Start the periodic task thread
                    self.periodic_thread.start()
                    self.start_periodic_task = True
                    break
                 
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
    
    def calculate_climb_rate(self):
        climb_max = 2.1
        climb_min = 1.5
        if(self.throttle > 0):
            self.throttle = 0.5 + (self.throttle/(climb_max * 2))
        else:
            self.throttle = 0.5 - (self.throttle/(climb_min * 2))

##################################################################################

def main():
    solving = one_opt()
    solving.run()

if __name__ == "__main__":
    main() 


# /uav1/mavros/local_position/odom
# /uav1/mavros/setpoint_raw/target_attitude

# rosbag record -O simulation_1 /uav1/mavros/local_position/odom /mavros/setpoint_raw/attitude /uav1/target_estimate /uav1/alpha_slack

# rosbag record -O simulation_1 /mavros/local_position/odom /mavros/setpoint_raw/attitude /uav1/target_estimate /uav1/alpha_slack /mavros/local_position/velocity_local

# roslaunch mavros apm.launch fcu_url:=/dev/ttyUSB0:921600

# sudo chmod 777 /dev/ttyUSB0

# ssh icarus@10.42.0.1

# icarus00

#
# rosbag record -O simulation_2 /mavros/local_position/odom /mavros/setpoint_raw/attitude /uav1/target_estimate /uav1/alpha_slack /mavros/local_position/velocity_local

#scp icarus@10.42.0.1:./.ros/sim_1.bag /home/joao/IRL_RESULTS_2/

#scp /home/joao/irl_tests/src/irl_tests/src/target_estimate.py icarus@10.42.0.1:joao_matias_ws/src/irl_tests/src
#scp /home/joao/irl_tests/src/irl_tests/src/target_estimate.py icarus@192.168.113.254:joao_matias_ws/src/irl_tests/src

#ssh icarus@192.168.113.168

#scp icarus@192.168.113.168:./.ros/sim_2.bag /home/joao/IRL_RESULTS_2/
