#!/bin/bash

# Default values
MAP_NUMBER=1
ROBOT_NUMBER=2
MODEL_EPISODE=200  # Use the model saved at episode 200
HEADLESS=true

# Parse command line arguments
for arg in "$@"; do
  case $arg in
    --map=*)
      MAP_NUMBER="${arg#*=}"
      ;;
    --robots=*)
      ROBOT_NUMBER="${arg#*=}"
      ;;
    --model=*)
      MODEL_EPISODE="${arg#*=}"
      ;;
    --headless=*)
      HEADLESS="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg"
      ;;
  esac
done

echo "Evaluating model with:"
echo "Map: $MAP_NUMBER"
echo "Robots: $ROBOT_NUMBER"
echo "Model from episode: $MODEL_EPISODE"
echo "Headless mode: $HEADLESS"

# Export environment variables for the ROS nodes
export map_number=$MAP_NUMBER
export robot_number=$ROBOT_NUMBER
export model_episode=$MODEL_EPISODE
export headless=$HEADLESS

# Launch the evaluation
cd ~/memoir/multi-robot-exploration-rl-master
source install/setup.bash
ros2 run start_reinforcement_learning run_mappo_evaluate --ros-args -p map_number:=$MAP_NUMBER -p robot_number:=$ROBOT_NUMBER -p model_episode:=$MODEL_EPISODE
