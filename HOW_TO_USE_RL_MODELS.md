# How to Use the Improved RL Model Training and Selection System

This guide explains how to use the enhanced reinforcement learning system for training robots to navigate to goals efficiently.

## Table of Contents
1. [Training Models](#training-models)
2. [Finding the Best Models](#finding-the-best-models)
3. [Evaluating Models](#evaluating-models)
4. [Understanding Early Stopping](#understanding-early-stopping)
5. [Tips for Model Selection](#tips-for-model-selection)

## Training Models

You can train models using either MAPPO or MADDPG algorithms.

### Training with MAPPO

```bash
# Basic training command
./train_improved_navigation.sh --map=1 --robots=2 --episodes=2000 --headless=true --fast=true

# Parameters:
# --map: Map number to use (1-5)
# --robots: Number of robots to train (1-5)
# --episodes: Maximum episodes to train (will stop early if goal success rate is high enough)
# --headless: Run without visualization (true/false)
# --fast: Run faster simulation (true/false)
```

### Training with MADDPG

```bash
# First set environment variables
export map_number=1
export robot_number=2
export algorithm=maddpg  # Important to specify maddpg here

# Launch the environment
ros2 launch start_rl_environment main.launch.py map_number:=$map_number robot_number:=$robot_number headless:=true fast_training:=true

# In a new terminal, launch the learning algorithm
ros2 launch start_reinforcement_learning start_learning.launch.py map_number:=$map_number robot_number:=$robot_number algorithm:=maddpg
```

Training will automatically stop when either:
1. The maximum number of episodes is reached
2. The goal success rate exceeds 85% for 20 consecutive evaluation intervals (default settings)

## Finding the Best Models

After training, use the utility script to find and analyze your models:

```bash
# List all trained models for a specific map and robot configuration
python3 src/start_reinforcement_learning/scripts/find_best_model.py --map=1 --robots=2 --list

# Show only the best model with its evaluation command
python3 src/start_reinforcement_learning/scripts/find_best_model.py --map=1 --robots=2 --best

# Evaluate a specific model by its key
python3 src/start_reinforcement_learning/scripts/find_best_model.py --map=1 --robots=2 --evaluate "best_score_150.50_ep345"
```

The script will display:
- Episode number
- Average score
- Goal success rate
- Date and time when the model was saved
- Directory path of the model
- Commands to run the model evaluation

## Evaluating Models

To evaluate a model:

1. **Start the environment**:
```bash
ros2 launch start_rl_environment main.launch.py map_number:=1 robot_number:=2 headless:=false fast_training:=false
```

2. **Evaluate a model** (use one of the following methods):

   a. **Best model for specific map and robots**:
   ```bash
   export map_number=1
   export robot_number=2
   export model_episode=0  # 0 means use best model
   ros2 launch start_reinforcement_learning evaluate_mappo.launch.py  # or evaluate_maddpg.launch.py
   ```

   b. **Specific periodic model**:
   ```bash
   export map_number=1
   export robot_number=2
   export model_episode=150  # Episode number of the periodic save
   ros2 launch start_reinforcement_learning evaluate_mappo.launch.py  # or evaluate_maddpg.launch.py
   ```

   c. **Direct path to model** (most reliable method):
   ```bash
   export map_number=1
   export robot_number=2
   export model_path="/absolute/path/to/model/directory"  # As shown by find_best_model.py
   ros2 launch start_reinforcement_learning evaluate_mappo.launch.py  # or evaluate_maddpg.launch.py
   ```

## Understanding Early Stopping

The early stopping mechanism prevents unnecessary training when the model is already performing well.

**It works by**:
1. Tracking goal success rate over recent episodes (last 50)
2. If the success rate stays above 85% (configurable) for 20 intervals
3. And at least 200 episodes have been completed
4. Then training stops and saves the best model

This ensures that you get models that reliably reach the goal without wasting computation time.

## Tips for Model Selection

When selecting models for deployment:

1. **Goal Success Rate**: This is the most important metric - it tells you how reliably the robot can reach the goal.

2. **Average Score**: Higher scores generally mean robots reach goals faster with fewer penalties.

3. **Evaluate Multiple Models**: Sometimes a model with slightly lower metrics might perform better in real-world situations.

4. **Check Latest Models**: If two models have similar metrics, the later one (higher episode number) is often more stable.

5. **Visual Verification**: Always visually verify the model's performance by watching it navigate in non-headless mode.

## Tracking Files

The system maintains JSON tracking files for all models:
```
src/start_reinforcement_learning/start_reinforcement_learning/deep_learning_weights/mappo/model_tracker_map1_robots2.json
src/start_reinforcement_learning/start_reinforcement_learning/deep_learning_weights/maddpg/model_tracker_map1_robots2.json
```

These files contain metadata about all saved models and can be inspected directly if needed.



# Step 1: Start the environment
source /opt/ros/humble/setup.bash
source /home/aladine/memoir/multi-robot-exploration-rl-master/install/setup.bash
ros2 launch start_rl_environment main.launch.py map_number:=1 robot_number:=3 headless:=false fast_training:=


# Step 2: Start the learning algorithm
source /opt/ros/humble/setup.bash
source /home/aladine/memoir/multi-robot-exploration-rl-master/install/setup.bash
ros2 launch start_reinforcement_learning start_learning.launch.py map_number:=1 robot_number:=3 algorithm:=mappo