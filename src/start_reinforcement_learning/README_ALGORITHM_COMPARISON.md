# MAPPO vs MADDPG Algorithm Comparison Framework

This framework provides tools to evaluate and compare the performance of MAPPO (Multi-Agent Proximal Policy Optimization) and MADDPG (Multi-Agent Deep Deterministic Policy Gradient) algorithms in multi-robot exploration tasks.

## Overview

The comparison framework runs both algorithms in the same environment with identical settings and collects comprehensive metrics on their performance. It then generates detailed analysis reports, visualizations, and statistical comparisons to help understand the strengths and weaknesses of each approach.

## Key Features

- **Side-by-side evaluation** of MAPPO and MADDPG algorithms
- **Comprehensive metrics collection** including rewards, steps, execution time, success rates, and more
- **Statistical analysis** of performance differences
- **Visualization** of learning curves and performance metrics
- **Detailed HTML reports** with algorithm comparisons and insights

## How to Run

### Using ROS2 Launch

The easiest way to run the comparison is using the provided launch file:

```bash
# Run with default parameters (map_number=1, robot_number=2, n_episodes=100)
ros2 launch start_reinforcement_learning compare_algorithms.launch.py

# Run with custom parameters
ros2 launch start_reinforcement_learning compare_algorithms.launch.py map_number:=2 robot_number:=3 n_episodes:=50
```

### Running Directly

You can also run the comparison directly:

```bash
# Set environment variables
export map_number=1
export robot_number=2
export n_episodes=100

# Run the comparison
ros2 run start_reinforcement_learning run_algorithm_comparison
```

## Output

The comparison framework generates several outputs in the `comparison_results` directory:

1. **CSV files** with raw metrics data
2. **Plot images** showing learning curves and performance comparisons
3. **HTML report** with comprehensive analysis and visualizations
4. **Summary statistics** highlighting key differences between algorithms

## Understanding the Results

### Key Metrics

- **Rewards**: Higher average rewards indicate better performance
- **Steps per Episode**: Fewer steps to complete tasks indicate more efficient exploration
- **Success Rate**: Percentage of episodes where robots reached the goal
- **Collision Rate**: Percentage of episodes where robots collided with obstacles
- **Execution Time**: Time taken to complete episodes

### Algorithm Differences

#### MAPPO (Multi-Agent Proximal Policy Optimization)
- Uses a centralized critic with decentralized actors (CTDE)
- Employs trust region optimization with clipped objective
- Utilizes Generalized Advantage Estimation (GAE)
- Typically more stable but may require more samples

#### MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
- Uses centralized training with decentralized execution
- Employs deterministic policy gradients
- Uses experience replay and target networks
- Often more sample-efficient but potentially less stable

## Customization

You can modify the comparison parameters by editing the launch file or setting environment variables. Key parameters include:

- `map_number`: The map to use for testing
- `robot_number`: Number of robots in the environment
- `n_episodes`: Number of episodes to run for each algorithm

## Requirements

- ROS2
- PyTorch
- NumPy
- Matplotlib
- Pandas
- SciPy

## Troubleshooting

- **Memory issues**: If you encounter memory problems, try reducing the number of episodes or robots
- **CUDA errors**: Make sure your PyTorch installation is compatible with your GPU
- **ROS2 errors**: Ensure all ROS2 dependencies are properly installed and sourced

## Extending the Framework

To add more algorithms to the comparison:

1. Implement the algorithm in a similar structure to MAPPO and MADDPG
2. Update the `algorithm_comparison.py` file to include the new algorithm
3. Modify the metrics analyzer to include the new algorithm in comparisons
