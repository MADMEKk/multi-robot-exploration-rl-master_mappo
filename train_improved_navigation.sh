#!/bin/bash

# Default values
MAP_NUMBER=1
ROBOT_NUMBER=5
EPISODES=2000
HEADLESS=true
FAST_TRAINING=true
ALGORITHM=mappo  # Default algorithm is MAPPO

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
    --algorithm=*)
      ALGORITHM="${arg#*=}"
      if [[ "$ALGORITHM" != "mappo" && "$ALGORITHM" != "maddpg" ]]; then
        echo "Error: Algorithm must be either 'mappo' or 'maddpg'"
        exit 1
      fi
      ;;
    --help)
      echo "Usage: ./train_improved_navigation.sh [options]"
      echo "Options:"
      echo "  --map=NUMBER         Map number to use (default: 1)"
      echo "  --robots=NUMBER      Number of robots (default: 5)"
      echo "  --episodes=NUMBER    Number of episodes to train (default: 2000)"
      echo "  --headless=BOOL      Run in headless mode (default: true)"
      echo "  --fast=BOOL          Run in fast training mode (default: true)"
      echo "  --algorithm=STRING   RL algorithm to use: mappo or maddpg (default: mappo)"
      echo "  --help               Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Use --help for usage information"
      ;;
  esac
done

echo "Starting improved navigation training with:"
echo "Map: $MAP_NUMBER"
echo "Robots: $ROBOT_NUMBER"
echo "Episodes: $EPISODES"
echo "Headless mode: $HEADLESS"
echo "Fast training: $FAST_TRAINING"
echo "Algorithm: $ALGORITHM"

# Export environment variables for the ROS nodes
export map_number=$MAP_NUMBER
export robot_number=$ROBOT_NUMBER
export episodes=$EPISODES
export headless=$HEADLESS
export fast_training=$FAST_TRAINING
export algorithm=$ALGORITHM

# Change to project directory
cd ~/memoir/multi-robot-exploration-rl-master
source install/setup.bash

# Step 1: Launch the environment
echo "Step 1: Launching the environment..."
ros2 launch start_rl_environment main.launch.py map_number:=$MAP_NUMBER robot_number:=$ROBOT_NUMBER headless:=$HEADLESS fast_training:=$FAST_TRAINING &
ENV_PID=$!

# Wait for the environment to initialize
echo "Waiting for environment to initialize (10 seconds)..."
sleep 10

# Step 2: Launch the learning algorithm
echo "Step 2: Launching the $ALGORITHM learning algorithm..."
ros2 launch start_reinforcement_learning start_learning.launch.py algorithm:=$ALGORITHM map_number:=$MAP_NUMBER robot_number:=$ROBOT_NUMBER

# If the learning algorithm terminates, also terminate the environment
kill $ENV_PID 2>/dev/null
