#!/bin/bash

# Unified script for launching both Gazebo environment and MAPPO training
# With options for headless mode and simulation speed

# Define parameters
MAP_NUMBER=1
ROBOT_NUMBER=3
CONFIG_PATH="src/start_reinforcement_learning/config/default_config.yaml"
HEADLESS=false
SIM_SPEED=1.0  # Simulation speed factor (1.0 = normal, 2.0 = 2x speed, etc.)

# Allow parameter overrides from command line
# Usage: ./train_run.sh [map_number] [robot_number] [config_path] [headless] [sim_speed]
if [ "$1" != "" ]; then
    MAP_NUMBER=$1
fi

if [ "$2" != "" ]; then
    ROBOT_NUMBER=$2
fi

if [ "$3" != "" ]; then
    CONFIG_PATH=$3
fi

if [ "$4" != "" ]; then
    HEADLESS=$4
fi

if [ "$5" != "" ]; then
    SIM_SPEED=$5
fi

echo "Starting training with map $MAP_NUMBER and $ROBOT_NUMBER robots"
echo "Using configuration from: $CONFIG_PATH"
echo "Headless mode: $HEADLESS"
echo "Simulation speed: ${SIM_SPEED}x"

# Setup ROS environment
source /opt/ros/humble/setup.bash
source install/setup.bash

# Convert headless boolean to ROS launch param
HEADLESS_PARAM=""
if [ "$HEADLESS" = true ]; then
    HEADLESS_PARAM="headless:=true gui:=false"
fi

# Launch Gazebo environment in the background
echo "Launching Gazebo environment..."
ros2 launch start_rl_environment rl_environment.launch.py map_number:=$MAP_NUMBER robot_number:=$ROBOT_NUMBER $HEADLESS_PARAM &
GAZEBO_PID=$!

# Wait for Gazebo to fully start (adjust time if needed)
echo "Waiting for Gazebo to initialize (15 seconds)..."
sleep 15

# Set simulation speed if different from 1.0
if (( $(echo "$SIM_SPEED != 1.0" | bc -l) )); then
    echo "Setting simulation speed to ${SIM_SPEED}x"
    
    # Method 1: Using gz service (for Gazebo Garden and newer)
    if command -v gz &> /dev/null; then
        echo "Using gz command to set simulation speed"
        gz service -s /world/rl_world/set_physics --reqtype ignition.msgs.Physics --reptype ignition.msgs.Boolean --timeout 1000 --req 'time_step:0.001 max_step_size:0.001 real_time_factor:'"$SIM_SPEED" || true
    fi
    
    # Method 2: Using ROS 2 parameter (for ROS 2 Humble and newer)
    echo "Using ROS parameter to set simulation speed"
    ros2 param set /gazebo real_time_update_rate "$(echo "$SIM_SPEED * 1000" | bc -l)" || true
fi

# Start the training
echo "Starting MAPPO training..."
ros2 run start_reinforcement_learning mappo_main \
  --map_number $MAP_NUMBER \
  --robot_number $ROBOT_NUMBER \
  --config $CONFIG_PATH \
  --tensorboard

# When training finishes, kill the Gazebo process
echo "Training complete, shutting down Gazebo..."
kill $GAZEBO_PID

# Alternatively, if you want to use terminals/tabs instead of background processes,
# uncomment the following and comment out the above method:

# Method 2: Using gnome-terminal (if you have gnome-terminal)
# echo "Launching Gazebo environment in a new terminal..."
# gnome-terminal -- bash -c "source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch start_rl_environment rl_environment.launch.py map_number:=$MAP_NUMBER robot_number:=$ROBOT_NUMBER; exec bash"
# 
# # Wait for Gazebo to fully start
# echo "Waiting for Gazebo to initialize (15 seconds)..."
# sleep 15
# 
# # Start the training in the current terminal
# echo "Starting MAPPO training..."
# ros2 run start_reinforcement_learning mappo_main \
#   --map_number $MAP_NUMBER \
#   --robot_number $ROBOT_NUMBER \
#   --config $CONFIG_PATH \
#   --tensorboard

echo "All processes finished" 