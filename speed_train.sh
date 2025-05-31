#!/bin/bash

# Script for running training in headless mode with optimized physics
# Author: Cascade AI Assistant

# Default parameters
MAP_NUMBER=1
ROBOT_NUMBER=2
ALGORITHM="mappo"
HEADLESS="true"
FAST_TRAINING="true"
SPEED_LEVEL="quality" # Options: quality, normal, ultra

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --map=*)
      MAP_NUMBER="${1#*=}"
      ;;
    --robots=*)
      ROBOT_NUMBER="${1#*=}"
      ;;
    --algorithm=*)
      ALGORITHM="${1#*=}"
      ;;
    --headless=*)
      HEADLESS="${1#*=}"
      ;;
    --fast=*)
      FAST_TRAINING="${1#*=}"
      ;;
    --speed=*)
      SPEED_LEVEL="${1#*=}"
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --map=N           Set map number (default: 1)"
      echo "  --robots=N        Set number of robots (default: 2)"
      echo "  --algorithm=NAME  Set algorithm: mappo or maddpg (default: mappo)"
      echo "  --headless=BOOL   Run in headless mode: true or false (default: true)"
      echo "  --fast=BOOL       Enable fast training: true or false (default: true)"
      echo "  --speed=LEVEL     Set speed level: quality, normal, or ultra (default: quality)"
      echo "  --help            Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
  shift
done

# Print configuration
echo "Starting training with the following configuration:"
echo "Map: $MAP_NUMBER"
echo "Robots: $ROBOT_NUMBER"
echo "Algorithm: $ALGORITHM"
echo "Headless mode: $HEADLESS"
echo "Fast training: $FAST_TRAINING"
echo "Speed level: $SPEED_LEVEL"
echo ""

# Set physics parameters based on speed level
if [ "$SPEED_LEVEL" = "ultra" ]; then
  # Configure ultra-fast simulation (10-20x speed) - prioritizes speed over accuracy
  echo "Configuring ultra-fast simulation (10-20x speed)..."
  
  # Set environment variables for ultra-fast mode
  export GAZEBO_PHYSICS_MAX_STEP_SIZE=0.004
  export GAZEBO_PHYSICS_RTF=20.0
  export GAZEBO_PHYSICS_UPDATE_RATE=2000
  
  # Start the environment in a separate terminal with ultra settings
  echo "Starting environment in ultra-fast mode..."
  gnome-terminal -- bash -c "ros2 launch start_rl_environment main.launch.py map_number:=$MAP_NUMBER robot_number:=$ROBOT_NUMBER headless:=$HEADLESS fast_training:=$FAST_TRAINING; exec bash"

elif [ "$SPEED_LEVEL" = "quality" ]; then
  # Configure quality-optimized simulation (5x speed) - balances speed and accuracy
  echo "Configuring quality-optimized simulation (5x speed with enhanced stability)..."
  
  # Set environment variables for quality-optimized mode
  export GAZEBO_PHYSICS_MAX_STEP_SIZE=0.002
  export GAZEBO_PHYSICS_RTF=5.0
  export GAZEBO_PHYSICS_UPDATE_RATE=1500
  export GAZEBO_PHYSICS_ODE_SOLVER_TYPE="quick"
  export GAZEBO_PHYSICS_ODE_SOLVER_ITERS=50
  
  # Start the environment with quality-optimized settings
  echo "Starting environment in quality-optimized mode..."
  gnome-terminal -- bash -c "ros2 launch start_rl_environment main.launch.py map_number:=$MAP_NUMBER robot_number:=$ROBOT_NUMBER headless:=$HEADLESS fast_training:=$FAST_TRAINING; exec bash"

else
  # Start the environment in a separate terminal with normal fast settings
  echo "Starting environment in normal fast mode..."
  gnome-terminal -- bash -c "ros2 launch start_rl_environment main.launch.py map_number:=$MAP_NUMBER robot_number:=$ROBOT_NUMBER headless:=$HEADLESS fast_training:=$FAST_TRAINING; exec bash"
fi

# Wait for environment to initialize
echo "Waiting for environment to initialize (5 seconds)..."
sleep 5

# Start the learning process
echo "Starting $ALGORITHM learning..."
ros2 launch start_reinforcement_learning start_learning.launch.py map_number:=$MAP_NUMBER robot_number:=$ROBOT_NUMBER algorithm:=$ALGORITHM
