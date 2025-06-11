#!/bin/bash

# Setup script for installing dependencies for multi-robot-exploration-rl

echo "Setting up dependencies for multi-robot exploration RL project..."

# Check if ROS 2 is installed
if ! command -v ros2 &> /dev/null; then
    echo "ERROR: ROS 2 is not installed or not in your PATH."
    echo "Please install ROS 2 Humble or newer first: https://docs.ros.org/en/humble/Installation.html"
    exit 1
fi

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in your PATH."
    exit 1
fi

# Create a Python virtual environment (optional)
read -p "Do you want to create a Python virtual environment? (y/n): " create_venv
if [[ "$create_venv" == "y" || "$create_venv" == "Y" ]]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Virtual environment activated."
fi

# Install Python dependencies
echo "Installing Python dependencies from requirements.txt..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Install additional ROS 2 packages
echo "Installing required ROS 2 packages..."
sudo apt-get update
sudo apt-get install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-xacro \
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher \
    ros-humble-tf2-ros \
    ros-humble-tf2-tools \
    python3-colcon-common-extensions

# Build the workspace
echo "Building the ROS 2 workspace..."
colcon build --symlink-install

# Setup automatic environment sourcing (optional)
if [ -f ~/.bashrc ]; then
    read -p "Do you want to automatically source this workspace in your .bashrc? (y/n): " source_bashrc
    if [[ "$source_bashrc" == "y" || "$source_bashrc" == "Y" ]]; then
        workspace_path=$(pwd)
        if ! grep -q "$workspace_path/install/setup.bash" ~/.bashrc; then
            echo "# Source multi-robot exploration RL workspace" >> ~/.bashrc
            echo "source $workspace_path/install/setup.bash" >> ~/.bashrc
            echo "Added workspace to .bashrc"
        else
            echo "Workspace already in .bashrc"
        fi
    fi
fi

echo ""
echo "Setup complete! Please follow these steps to start using the project:"
echo "1. Source the workspace: source install/setup.bash"
echo "2. Run the training script: ./train_run.sh"
echo "   For headless mode with increased simulation speed: ./train_run.sh 1 3 src/start_reinforcement_learning/config/default_config.yaml true 10.0"
echo ""
echo "For more options and configurations, check the README and configuration files." 