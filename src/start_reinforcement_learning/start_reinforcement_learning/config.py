"""
Configuration management for the reinforcement learning system.
This module defines all the hyperparameters and environment settings in one place.
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional
import yaml


@dataclass
class EnvironmentConfig:
    """Configuration for the simulation environment."""
    map_number: int = 1
    number_of_robots: int = 3
    max_steps_per_episode: int = 500
    goal_radius: float = 0.5
    
    # Robot velocity constraints
    max_linear_vel: float = 0.6
    min_linear_vel: float = 0.05
    max_angular_vel: float = 0.5
    min_angular_vel: float = -0.5
    
    # Rewards
    goal_reward: float = 20.0
    collision_reward: float = -20.0
    time_penalty: float = -0.1
    movement_reward: float = 0.2
    slow_movement_penalty: float = -0.5
    excessive_rotation_penalty: float = -0.2
    
    # Exploration settings
    exploration_reward_scale: float = 0.5
    individual_exploration_scale: float = 0.2
    max_acceptable_overlap: float = 0.2  # Percentage of overlap considered acceptable
    overlap_penalty_scale: float = 0.3
    
    # Map dimensions used for exploration tracking
    map_size: Tuple[float, float] = (20.0, 20.0)  # (width, height) in meters
    exploration_grid_resolution: float = 0.5  # Cell size in meters


@dataclass
class AlgorithmConfig:
    """Configuration for the MAPPO algorithm."""
    # Network architecture
    actor_hidden_dim_1: int = 512
    actor_hidden_dim_2: int = 512
    critic_hidden_dim_1: int = 512
    critic_hidden_dim_2: int = 512
    
    # PPO hyperparameters
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda parameter
    clip_param: float = 0.2  # PPO clip parameter
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy coefficient
    
    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    lr_decay_rate: float = 0.9999  # Learning rate decay per episode
    min_lr: float = 1e-5  # Minimum learning rate
    
    # Buffer
    buffer_size: int = 1000000  # Replay buffer size
    batch_size: int = 2048
    max_episodes_in_buffer: int = 50
    
    # Training
    update_epochs: int = 4  # Number of epochs to update policy per batch
    max_grad_norm: float = 0.5  # Gradient clipping
    

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str = "default_experiment"
    num_episodes: int = 10000
    evaluate_interval: int = 100
    save_interval: int = 50
    print_interval: int = 10
    
    # Training metrics
    metrics_dir: str = "training_metrics"
    plot_interval: int = 50
    
    # Checkpoint settings
    checkpoint_dir: str = "deep_learning_weights/mappo"
    save_best_only: bool = False


class ConfigManager:
    """Manager class for configuration settings."""
    
    def __init__(self, 
                 env_config: Optional[EnvironmentConfig] = None,
                 alg_config: Optional[AlgorithmConfig] = None,
                 exp_config: Optional[ExperimentConfig] = None):
        """
        Initialize the configuration manager.
        
        Args:
            env_config: Environment configuration
            alg_config: Algorithm configuration
            exp_config: Experiment configuration
        """
        self.env = env_config or EnvironmentConfig()
        self.alg = alg_config or AlgorithmConfig()
        self.exp = exp_config or ExperimentConfig()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ConfigManager':
        """
        Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            ConfigManager instance with loaded configuration
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Parse environment config
        env_data = config_data.get('environment', {})
        env_config = EnvironmentConfig(**env_data)
        
        # Parse algorithm config
        alg_data = config_data.get('algorithm', {})
        alg_config = AlgorithmConfig(**alg_data)
        
        # Parse experiment config
        exp_data = config_data.get('experiment', {})
        exp_config = ExperimentConfig(**exp_data)
        
        return cls(env_config, alg_config, exp_config)
    
    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            yaml_path: Path where to save the YAML configuration
        """
        config_data = {
            'environment': self._dataclass_to_dict(self.env),
            'algorithm': self._dataclass_to_dict(self.alg),
            'experiment': self._dataclass_to_dict(self.exp),
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    def update_from_cmd_args(self, args: Dict[str, Any]) -> None:
        """
        Update configuration based on command line arguments.
        
        Args:
            args: Dictionary of argument name to value
        """
        # Update environment config
        for key, value in args.items():
            if hasattr(self.env, key):
                setattr(self.env, key, value)
            elif hasattr(self.alg, key):
                setattr(self.alg, key, value)
            elif hasattr(self.exp, key):
                setattr(self.exp, key, value)
    
    @staticmethod
    def _dataclass_to_dict(dataclass_instance: Any) -> Dict[str, Any]:
        """Convert a dataclass instance to a dictionary."""
        return {field: getattr(dataclass_instance, field) 
                for field in dataclass_instance.__dataclass_fields__}


# Default configuration instance
default_config = ConfigManager()

def get_config() -> ConfigManager:
    """Get the default configuration instance."""
    return default_config 