#!/usr/bin/env python

import rospy
import subprocess
from mrs_msgs.srv import Vec4


def run_command(command):
    rospy.loginfo(f"Running command: {command}")
    subprocess.Popen(command, shell=True)

if __name__ == "__main__":
    rospy.init_node('single_uav_sequence')

    # Step 3: Run UAV scripts
    run_command("rosrun irl_tests target_estimate.py")

    rospy.sleep(5)

    sim_number = 2
    bag1_name = "sim_" + str(sim_number)

    # Step 5: Start recording rosbag files
    run_command("rosbag record -q -O " + bag1_name + " /mavros/local_position/odom /mavros/setpoint_raw/attitude /uav1/target_estimate /uav1/alpha_slack /mavros/local_position/velocity_local >/dev/null 2>&1")

    # Step 6: Wait for 5 seconds
    rospy.sleep(5)

    # Step 7: Run target publishing script
    run_command("rosrun irl_tests target_pub.py")

    rospy.spin()
