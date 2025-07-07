#!/usr/bin/env python

import rospy
import subprocess
import time
from std_srvs.srv import Empty
from mrs_msgs.srv import Vec4
from geometry_msgs.msg import Vector3
import signal
import os
import rosnode


def call_goto_service(uav_name, x, y, z, yaw):
    service_name = f'/{uav_name}/control_manager/goto'
    rospy.wait_for_service(service_name)
    try:
        goto_service = rospy.ServiceProxy(service_name, Vec4)
        # Send the entire goal as a list
        goal = [x, y, z, yaw]
        goto_service(goal)  # Call the service with the single 'goal' argument
        rospy.loginfo(f"Sent goto command to {uav_name}: {goal}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed for {uav_name}: {e}")

def run_command(command):
    rospy.loginfo(f"Running command: {command}")
    subprocess.Popen(command, shell=True)

if __name__ == "__main__":
    rospy.init_node('multi_uav_sequence')

    # Step 1: Send GOTO commands to 3 UAVs
    x_offset = 0
    y_offset = 0

    rospy.sleep(30)
    call_goto_service('uav1', 0 + x_offset, -10 + y_offset - 20, 8.6, 1.57)
    rospy.sleep(10)
    call_goto_service('uav3', -8 + x_offset - 15, 4.77 + y_offset, 8.6, -0.52)
    rospy.sleep(10)
    call_goto_service('uav2', 8 + x_offset + 15, 4.77 + y_offset, 8.6, -2.09)


    # Step 2: Wait for 40 seconds
    rospy.sleep(10)

    # Step 1: Send GOTO commands to 3 UAVs
    call_goto_service('uav1', 0 + x_offset, -10 + y_offset, 8.6, 1.57)
    call_goto_service('uav2', 8 + x_offset, -4.77 + y_offset, 8.6, 2.09) #-2.09
    call_goto_service('uav3', -8 + x_offset, -4.77 + y_offset, 8.6, 0.52) #-0.52

    # Step 2: Wait for 40 seconds
    rospy.sleep(15)

    # Step 3: Run UAV scripts
    run_command("rosrun multi_drone 1_uav.py")
    run_command("rosrun multi_drone 2_uav.py")
    run_command("rosrun multi_drone 3_uav.py")

    rospy.sleep(5)

    run_command("gz physics -u 200")

    # Step 4: Wait for 5 seconds
    rospy.sleep(5)

    sim_number = 6
    bag1_name = "simulation_1_" + str(sim_number)
    bag2_name = "simulation_2_" + str(sim_number)
    bag3_name = "simulation_3_" + str(sim_number)

    # Step 5: Start recording rosbag files
    run_command("rosbag record -q -O /home/joao/multi_drone_ws/uav1/" + bag1_name + " /uav1/estimation_manager/uav_state /uav1/hw_api/attitude_cmd /uav1/target_estimate /uav1/alpha_slack >/dev/null 2>&1")
    run_command("rosbag record -q -O /home/joao/multi_drone_ws/uav2/" + bag2_name + " /uav2/estimation_manager/uav_state /uav2/hw_api/attitude_cmd /uav1/target_estimate /uav2/alpha_slack >/dev/null 2>&1")
    run_command("rosbag record -q -O /home/joao/multi_drone_ws/uav3/" + bag3_name + " /uav3/estimation_manager/uav_state /uav3/hw_api/attitude_cmd /uav1/target_estimate /uav3/alpha_slack >/dev/null 2>&1")

    # Step 6: Wait for 5 seconds
    rospy.sleep(5)

    # Step 7: Run target publishing script
    run_command("rosrun multi_drone target_pub.py")

    rospy.spin()
