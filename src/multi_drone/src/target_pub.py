#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
import time  # For delay purposes if needed
import os
import rospkg  # ROS package library to find the package path
import subprocess

def stop_rosbag():
    # Send a SIGINT to all rosbag processes
    try:
        subprocess.call(['pkill', '-2', 'rosbag'])  # -2 is SIGINT (Ctrl+C)
        print("Rosbag recording stopped.")
    except Exception as e:
        print(f"Failed to stop rosbag: {str(e)}")

def kill_roslaunch_and_roscore():
    """Kill roslaunch and roscore processes using pkill."""
    try:
        # Kill all roslaunch processes
        subprocess.call(['pkill', '-f', 'roslaunch'])
        rospy.loginfo("All roslaunch processes killed.")

        # Kill all roscore processes
        # subprocess.call(['pkill', '-f', 'roscore'])
        # rospy.loginfo("All roscore processes killed.")
    except Exception as e:
        rospy.logerr(f"Failed to kill roslaunch or roscore processes: {e}")


def read_data_from_file(filename):
    """
    Reads the position and velocity data from a text file.
    The file should be in the format: position_x, position_y, velocity_x, velocity_y
    """
    data_list = []
    # Initialize ROS package manager
    rospack = rospkg.RosPack()

    # Get the path of your ROS package
    package_path = rospack.get_path('multi_drone')  # Replace 'my_package' with your package name

    # Construct the full path to the TXT file in the config folder
    txt_file_path = os.path.join(package_path, 'config', filename)

    # Open and read the TXT file
    with open(txt_file_path, 'r') as file:
        for line in file:
            # Strip the newline character and split the line by commas
            values = line.strip().split(',')
            # Convert each value from string to float and append it as a list to data_list
            data_list.append([float(val) for val in values])
    return data_list

def talker():
    rospy.init_node('target_publisher', anonymous=True)
    pub = rospy.Publisher('/uav1/target_estimate', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(4)  # 5 Hz - passar para 4 com multi_drone

    # Read data from the text file (change the file path accordingly)
    filename = "target_trajectory.txt"
    data = read_data_from_file(filename)

    index = 0

    while not rospy.is_shutdown():
        # Ensure we don't go beyond the available data in the list
        if index >= len(data):
            stop_rosbag()
            rospy.signal_shutdown("Finished execution, shutting down target_pub.")
            kill_roslaunch_and_roscore()
            # index = 0  # Optionally, loop back to the start
            

        # Retrieve the current position and velocity
        current_data = data[index]

        # Construct the message
        data_info = Float64MultiArray()
        data_info.data = current_data

        # Publish the message
        # print(data_info)
        pub.publish(data_info)

        # Increment the index to get the next line of data in the next iteration
        index += 1

        # Sleep to maintain the loop rate
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass