#!/bin/bash

# Grant permissions to the USB port
sudo chmod 777 /dev/ttyUSB0

# Start the mavros node
roslaunch mavros apm.launch fcu_url:=/dev/ttyUSB0:921600 &
