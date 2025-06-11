# Multi-Robot Exploration with MAPPO

This repository contains an improved implementation of Multi-Agent Proximal Policy Optimization (MAPPO) for multi-robot exploration in unknown environments. The project uses ROS 2 and Gazebo for simulation and training.

## 🚀 Key Features

- **Multi-Agent Reinforcement Learning**: Uses MAPPO to train multiple robots simultaneously
- **Enhanced Reward Function**: Improved reward design to encourage better exploration
- **Configuration Management**: Flexible configuration system for experiment reproducibility
- **Memory-Efficient Implementation**: Optimized buffer implementation for better performance
- **TensorBoard Integration**: Advanced visualization of training metrics
- **Headless Training**: Support for accelerated training without GUI
- **Variable Simulation Speed**: Control simulation speed for faster training

## 📋 Requirements

- Ubuntu 22.04 or newer
- ROS 2 Humble or newer
- Python 3.8+
- NVIDIA GPU (recommended for faster training)

## 🛠️ Installation

1. **Install ROS 2 Humble**:
   Follow the [official ROS 2 installation instructions](https://docs.ros.org/en/humble/Installation.html).

2. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/multi-robot-exploration-rl.git
   cd multi-robot-exploration-rl
   ```

3. **Install dependencies**:
   ```bash
   ./setup_dependencies.sh
   ```
   
4. **Build the workspace**:
   ```bash
   colcon build --symlink-install
   ```

5. **Source the workspace**:
   ```bash
   source install/setup.bash
   ```

## 🎮 Usage

### Standard Training

Run training with the default configuration:

```bash
./train_run.sh
```

This will:
1. Launch Gazebo with the default environment
2. Initialize robots with specified parameters
3. Start the MAPPO training
4. Record results and visualize on TensorBoard

### Advanced Training Options

Run with custom parameters:

```bash
./train_run.sh [map_number] [robot_number] [config_path] [headless] [sim_speed]
```

- `map_number`: Map configuration to use (default: 1)
- `robot_number`: Number of robots (default: 3)
- `config_path`: Path to configuration file (default: src/start_reinforcement_learning/config/default_config.yaml)
- `headless`: Run without GUI (true/false, default: false)
- `sim_speed`: Simulation speed factor (default: 1.0, higher values = faster simulation)

### Examples

#### Training with 5 robots on map 2:
```bash
./train_run.sh 2 5
```

#### Headless training with 10x simulation speed:
```bash
./train_run.sh 1 3 src/start_reinforcement_learning/config/default_config.yaml true 10.0
```

## ⚙️ Configuration

The project uses a structured configuration system for better reproducibility. Main configuration files are located at:

```
src/start_reinforcement_learning/config/
```

### Creating Custom Configurations

Copy the default config and modify it for your needs:

```bash
cp src/start_reinforcement_learning/config/default_config.yaml src/start_reinforcement_learning/config/my_custom_config.yaml
```

Edit parameters in your custom config file and use it for training:

```bash
./train_run.sh 1 3 src/start_reinforcement_learning/config/my_custom_config.yaml
```

## 📊 Monitoring Training

### TensorBoard

Training progress is automatically logged to TensorBoard. Access the dashboard at:

```
http://localhost:6006
```

If TensorBoard is not running, start it with:

```bash
tensorboard --logdir=src/start_reinforcement_learning/tensorboard_logs
```

### Training Metrics

Additional training metrics are saved to:

```
src/start_reinforcement_learning/training_metrics/
```

This includes:
- CSV files with training statistics
- Configuration snapshots
- Performance plots

## 🧠 Model Architecture

The implementation uses a state-of-the-art MAPPO architecture with:

- Actor-Critic networks for each agent
- Centralized training with decentralized execution
- Generalized Advantage Estimation (GAE)
- Learning rate scheduling
- Entropy regularization
- Value function clipping

## 🔍 Key Improvements

1. **Learning Rate Scheduling**: Exponential decay for more stable training
2. **Enhanced Reward Function**: Better exploration incentives and obstacle avoidance
3. **Memory-Efficient Buffer**: Optimized for large-scale multi-agent training
4. **Configuration Management**: Comprehensive config system for experiment tracking
5. **TensorBoard Integration**: Advanced visualization for debugging and analysis
6. **Headless Training**: Support for compute-efficient training

## 📝 Citation

If you use this code in your research, please cite:

```
@misc{multi-robot-exploration-rl,
  author = {Your Name},
  title = {Multi-Robot Exploration with MAPPO},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/your-username/multi-robot-exploration-rl}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 