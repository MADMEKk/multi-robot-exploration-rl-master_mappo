#!/bin/bash

# Default values
MAP_NUMBER=1
ROBOT_NUMBER=5
EPISODES=2000
HEADLESS=true
FAST_TRAINING=true

# Parse command line arguments
for arg in "$@"; do
  case $arg in
    --map=*)
      MAP_NUMBER="${arg#*=}"
      ;;
    --robots=*)
      ROBOT_NUMBER="${arg#*=}"
      ;;
    --episodes=*)
      EPISODES="${arg#*=}"
      ;;
    --headless=*)
      HEADLESS="${arg#*=}"
      ;;
    --fast=*)
      FAST_TRAINING="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg"
      ;;
  esac
done

echo "Starting improved navigation training with:"
echo "Map: $MAP_NUMBER"
echo "Robots: $ROBOT_NUMBER"
echo "Episodes: $EPISODES"
echo "Headless mode: $HEADLESS"
echo "Fast training: $FAST_TRAINING"

# Export environment variables for the ROS nodes
export map_number=$MAP_NUMBER
export robot_number=$ROBOT_NUMBER
export episodes=$EPISODES
export headless=$HEADLESS
export fast_training=$FAST_TRAINING

# Launch the training
cd ~/memoir/multi-robot-exploration-rl-master
source install/setup.bash
ros2 launch start_reinforcement_learning start_learning.launch.py algorithm:=mappo map_number:=$MAP_NUMBER robot_number:=$ROBOT_NUMBER
