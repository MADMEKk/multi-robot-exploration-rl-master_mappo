#!/bin/bash

# New speed training script for improved MAPPO implementation
# This script assumes Gazebo with the robot environment is already running

# Define parameters
MAP_NUMBER=1
ROBOT_NUMBER=3
CONFIG_PATH="src/start_reinforcement_learning/config/default_config.yaml"

# Allow parameter overrides from command line
# Usage: ./new_speed_train.sh [map_number] [robot_number] [config_path]
if [ "$1" != "" ]; then
    MAP_NUMBER=$1
fi

if [ "$2" != "" ]; then
    ROBOT_NUMBER=$2
fi

if [ "$3" != "" ]; then
    CONFIG_PATH=$3
fi

echo "Starting training with map $MAP_NUMBER and $ROBOT_NUMBER robots"
echo "Using configuration from: $CONFIG_PATH"

# Setup ROS environment
source /opt/ros/humble/setup.bash
source install/setup.bash

# Run the training with the new implementation
ros2 run start_reinforcement_learning mappo_main \
  --map_number $MAP_NUMBER \
  --robot_number $ROBOT_NUMBER \
  --config $CONFIG_PATH \
  --tensorboard

echo "Training complete" 