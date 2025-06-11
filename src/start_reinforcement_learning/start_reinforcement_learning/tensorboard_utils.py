"""
TensorBoard integration utilities for visualizing training metrics.
"""

import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TensorboardLogger:
    """Class for logging training metrics to TensorBoard."""
    
    def __init__(self, log_dir: str, experiment_name: Optional[str] = None, map_number: int = 1, robot_number: int = 3):
        """
        Initialize the TensorBoard logger.
        
        Args:
            log_dir: Base directory for TensorBoard logs
            experiment_name: Name of the experiment
            map_number: Map number being used
            robot_number: Number of robots in the environment
        """
        if not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard not available. Install torch and tensorboard packages for TensorBoard logging.")
            self.writer = None
            return
        
        # Create a unique run name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            run_name = f"{experiment_name}_map{map_number}_robots{robot_number}_{timestamp}"
        else:
            run_name = f"mappo_map{map_number}_robots{robot_number}_{timestamp}"
        
        # Create full log directory path
        full_log_dir = os.path.join(log_dir, run_name)
        os.makedirs(full_log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=full_log_dir)
        print(f"TensorBoard logs will be saved to: {full_log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value to TensorBoard.
        
        Args:
            tag: Name of the metric
            value: Value to log
            step: Training step or episode number
        """
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_multiple_scalars(self, tag_prefix: str, value_dict: Dict[str, float], step: int):
        """
        Log multiple scalar values with the same prefix to TensorBoard.
        
        Args:
            tag_prefix: Prefix for the metric names
            value_dict: Dictionary mapping metric names to values
            step: Training step or episode number
        """
        if self.writer is not None:
            for name, value in value_dict.items():
                full_tag = f"{tag_prefix}/{name}" if tag_prefix else name
                self.writer.add_scalar(full_tag, value, step)
    
    def log_episode_results(self, episode: int, score: float, avg_score: float, steps: int,
                           success: bool, collision: bool, timeout: bool,
                           exploration_coverage: float, learning_rate: float):
        """
        Log comprehensive episode results to TensorBoard.
        
        Args:
            episode: Episode number
            score: Episode score
            avg_score: Average score over recent episodes
            steps: Number of steps in the episode
            success: Whether the episode ended in success
            collision: Whether the episode ended in collision
            timeout: Whether the episode ended in timeout
            exploration_coverage: Environment exploration coverage
            learning_rate: Current learning rate
        """
        if self.writer is None:
            return
            
        # Log basic episode metrics
        self.writer.add_scalar('Episode/Score', score, episode)
        self.writer.add_scalar('Episode/AvgScore', avg_score, episode)
        self.writer.add_scalar('Episode/Steps', steps, episode)
        
        # Log episode outcome (one-hot)
        outcome_dict = {
            'Success': 1.0 if success else 0.0,
            'Collision': 1.0 if collision else 0.0,
            'Timeout': 1.0 if timeout else 0.0,
        }
        self.log_multiple_scalars('Outcome', outcome_dict, episode)
        
        # Log exploration metrics
        self.writer.add_scalar('Exploration/Coverage', exploration_coverage, episode)
        
        # Log training parameters
        self.writer.add_scalar('Parameters/LearningRate', learning_rate, episode)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int, bins: str = 'auto'):
        """
        Log a histogram of values to TensorBoard.
        
        Args:
            tag: Name of the metric
            values: Array of values to log
            step: Training step or episode number
            bins: Number of bins for the histogram
        """
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step, bins=bins)
    
    def log_agent_metrics(self, episode: int, agent_idx: int, policy_loss: float, 
                         value_loss: float, entropy_loss: float):
        """
        Log per-agent training metrics.
        
        Args:
            episode: Episode number
            agent_idx: Agent index
            policy_loss: Policy loss value
            value_loss: Value function loss value
            entropy_loss: Entropy loss value
        """
        if self.writer is None:
            return
            
        tag_prefix = f"Agent{agent_idx}"
        metrics = {
            'PolicyLoss': policy_loss,
            'ValueLoss': value_loss,
            'EntropyLoss': entropy_loss
        }
        self.log_multiple_scalars(tag_prefix, metrics, episode)
    
    def log_text(self, tag: str, text_string: str, step: int):
        """
        Log text to TensorBoard.
        
        Args:
            tag: Name for the text
            text_string: Text to log
            step: Training step or episode number
        """
        if self.writer is not None:
            self.writer.add_text(tag, text_string, step)
    
    def log_rewards_breakdown(self, episode: int, reward_components: Dict[str, float]):
        """
        Log breakdown of reward components.
        
        Args:
            episode: Episode number
            reward_components: Dictionary mapping reward component names to values
        """
        if self.writer is not None:
            self.log_multiple_scalars('Rewards', reward_components, episode)
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


def test_tensorboard_logger():
    """Simple test function for the TensorBoard logger."""
    if not TENSORBOARD_AVAILABLE:
        print("TensorBoard not available. Skipping test.")
        return
        
    logger = TensorboardLogger("./logs", "test_experiment", map_number=1, robot_number=3)
    
    # Log some test data
    for episode in range(100):
        # Simulate some metrics
        score = np.random.normal(0, 1) * episode / 10
        avg_score = score * 0.9
        steps = np.random.randint(50, 200)
        success = np.random.random() > 0.7
        collision = False if success else np.random.random() > 0.6
        timeout = not (success or collision)
        exploration = min(1.0, episode / 100 + np.random.random() * 0.1)
        lr = 0.001 * (0.9 ** (episode // 10))
        
        # Log the metrics
        logger.log_episode_results(
            episode, score, avg_score, steps, success, collision, timeout, exploration, lr
        )
        
        # Log some agent-specific metrics
        for agent_idx in range(3):
            policy_loss = np.random.normal(0, 1) * 0.1
            value_loss = np.random.normal(0, 1) * 0.2
            entropy_loss = np.random.normal(0, 1) * 0.05
            logger.log_agent_metrics(episode, agent_idx, policy_loss, value_loss, entropy_loss)
    
    logger.close()
    print("TensorBoard test complete. Run 'tensorboard --logdir=./logs' to view results.")


if __name__ == "__main__":
    test_tensorboard_logger() 