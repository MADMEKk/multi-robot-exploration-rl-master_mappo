import os
import rclpy
from rclpy.node import Node
import numpy as np
import csv
from datetime import datetime
import time
import argparse

from start_reinforcement_learning.logic import Env
from start_reinforcement_learning.mappo_algorithm.mappo import MAPPO
from start_reinforcement_learning.mappo_algorithm.efficient_buffer import EfficientMultiAgentBuffer
from start_reinforcement_learning.config import ConfigManager, get_config
from start_reinforcement_learning.tensorboard_utils import TensorboardLogger, TENSORBOARD_AVAILABLE
import torch as T
import gc
from ament_index_python.packages import get_package_share_directory
import matplotlib.pyplot as plt
import yaml

# Convert list of arrays to one flat array of observations
def obs_list_to_state_vector(observation):
    state = np.array([])    
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

# Main function that runs the MAPPO algorithm
class MAPPONode(Node):
    def __init__(self, config_path=None):
        super().__init__('mappo_node')

        # Initialize config - load from file if provided, otherwise use defaults
        if config_path and os.path.exists(config_path):
            self.config = ConfigManager.from_yaml(config_path)
            self.get_logger().info(f"Loaded configuration from {config_path}")
        else:
            self.config = get_config()
            self.get_logger().info("Using default configuration")

        # Access the parameters passed from the launch file or command line
        map_number = self.declare_parameter('map_number', self.config.env.map_number).get_parameter_value().integer_value
        robot_number = self.declare_parameter('robot_number', self.config.env.number_of_robots).get_parameter_value().integer_value
        
        # Update config with ROS parameters
        self.config.env.map_number = map_number
        self.config.env.number_of_robots = robot_number
        
        self.get_logger().info(f"Map number: {map_number}")
        self.get_logger().info(f"Robot number: {robot_number}")

        # Set environment with action size
        env = Env(robot_number, map_number)
        self.get_logger().info(f"Map number: {map_number}")
        n_agents = env.number_of_robots
        
        actor_dims = env.observation_space()
        critic_dims = sum(actor_dims)

        # Action space is discrete, one of 9 actions (3 linear x 3 angular)
        n_actions = 9  # 3 linear velocity options x 3 angular velocity options

        # Use direct path instead of get_package_share_directory which might fail
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Include map number and robot number in the checkpoint directory path
        chkpt_dir_var = os.path.join(base_path, 'start_reinforcement_learning', 
                                    self.config.exp.checkpoint_dir, 
                                    f'map{map_number}_robots{robot_number}')
        
        self.get_logger().info(f"Checkpoint directory: {chkpt_dir_var}")
        
        # Create training metrics directory
        training_metrics_dir = os.path.join(base_path, 'start_reinforcement_learning', 
                                          self.config.exp.metrics_dir)
        os.makedirs(training_metrics_dir, exist_ok=True)
        
        # Save the current configuration
        config_save_path = os.path.join(training_metrics_dir, f"config_map{map_number}_robots{robot_number}.yaml")
        self.config.to_yaml(config_save_path)
        self.get_logger().info(f"Saved configuration to: {config_save_path}")
        
        # Initialize TensorBoard logger
        tensorboard_log_dir = os.path.join(base_path, 'start_reinforcement_learning', 'tensorboard_logs')
        self.tb_logger = TensorboardLogger(
            tensorboard_log_dir, 
            experiment_name=self.config.exp.name,
            map_number=map_number,
            robot_number=robot_number
        )
        if TENSORBOARD_AVAILABLE:
            self.get_logger().info(f"TensorBoard logging enabled")
        else:
            self.get_logger().warning(f"TensorBoard not available. Install torch and tensorboard packages for enhanced visualizations.")
            
        # Timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize main algorithm with config parameters
        mappo_agents = MAPPO(actor_dims, critic_dims, n_agents, n_actions, 
                             fc1=self.config.alg.actor_hidden_dim_1, 
                             fc2=self.config.alg.actor_hidden_dim_2, 
                             tau=0.00025,
                             alpha=self.config.alg.actor_lr, 
                             beta=self.config.alg.critic_lr, 
                             gamma=self.config.alg.gamma,
                             gae_lambda=self.config.alg.gae_lambda,
                             clip_param=self.config.alg.clip_param,
                             entropy_coef=self.config.alg.entropy_coef,
                             value_coef=self.config.alg.value_coef,
                             scenario='robot',
                             chkpt_dir=chkpt_dir_var, 
                             node_logger=self)

        # Initialize memory using the efficient buffer implementation
        memory = EfficientMultiAgentBuffer(
            self.config.alg.buffer_size, 
            critic_dims, 
            actor_dims,
            n_actions, 
            n_agents, 
            batch_size=self.config.alg.batch_size,
            gamma=self.config.alg.gamma, 
            gae_lambda=self.config.alg.gae_lambda)

        # Training parameters from config
        PRINT_INTERVAL = self.config.exp.print_interval
        N_GAMES = self.config.exp.num_episodes
        total_steps = 0
        score_history = []
        step_history = []
        success_history = []
        collision_history = []
        timeout_history = []
        exploration_history = []
        best_score = -20

        # Create training metrics file
        metrics_filename = os.path.join(training_metrics_dir, 
                                     f"mappo_training_map{map_number}_robots{robot_number}_{timestamp}.csv")
        with open(metrics_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Score', 'Steps', 'Success', 'Collision', 'Timeout', 
                            'Exploration_Coverage', 'Avg_Reward', 'Learning_Rate'])

        # Evaluate flag - for testing only
        evaluate = False
        if evaluate:
            mappo_agents.load_checkpoint()

        # Training loop
        for i in range(N_GAMES):
            # Set a random goal position for training episodes
            # This helps the model learn to navigate to arbitrary goals
            if i > 0 and i % 5 == 0:  # Change goal every 5 episodes (can adjust frequency)
                # Generate random goal coordinates within the map boundaries
                # Adjust these ranges based on your map size
                if map_number == 1:
                    goal_x = np.random.uniform(-1.5, 1.5)  # More limited x-range (was -2.0 to 2.5)
                    goal_y = np.random.uniform(-5.0, -1.0)  # More limited y-range (was -10.0 to -1.0)
                else:
                    # Map 2 boundaries - MODIFIED to more reasonable ranges
                    goal_x = np.random.uniform(-3.0, 5.0)  # More limited x-range (was -5.0 to 9.0)
                    goal_y = np.random.uniform(-5.0, -1.0) 
                
                try:
                    # Set the new goal position
                    env.set_goal(goal_x, goal_y)
                    self.get_logger().info(f"Set new random goal at ({goal_x:.2f}, {goal_y:.2f}) for episode {i}")
                except Exception as e:
                    self.get_logger().error(f"Failed to set random goal: {e}")
            
            # Reset to get initial observation
            obs = env.reset()
            # Convert dict -> list of arrays
            list_obs = list(obs.values())
            score = 0
            done = [False] * n_agents
            terminal = [False] * n_agents
            episode_step = 0
            
            # Episode tracking variables
            episode_success = False
            episode_collision = False
            episode_timeout = False
            
            # Truncated means episode has reached max number of steps, done means collided or reached goal
            while not any(terminal):
                # Convert list of observations to global state
                global_state = obs_list_to_state_vector(list_obs)
                
                # Get the actions that the algorithm thinks are best in given observation (decentralized execution)
                actions, values, log_probs = mappo_agents.choose_action(obs, global_state)
                
                # Use step function to get next state and reward info as well as if the episode is 'done'
                obs_, reward, done, truncated, info = env.step(actions)
                
                # Convert dict -> list of arrays
                list_done = list(done.values())
                list_reward = list(reward.values())
                list_obs_ = list(obs_.values())
                list_trunc = list(truncated.values())
                
                # Convert list of arrays to one flat array of observations (global state)
                global_state_ = obs_list_to_state_vector(list_obs_)
                
                # Check if episode is done
                terminal = [d or t for d, t in zip(list_done, list_trunc)]

                # Check for goal or collision
                if any(list_done):
                    # Check if any robot reached the goal
                    for robot_idx in range(n_agents):
                        if env.hasReachedGoal(list_obs[robot_idx], robot_idx):
                            episode_success = True
                            break
                    
                    # If not goal, must be collision
                    if not episode_success:
                        episode_collision = True
                        
                # Check for timeout
                if any(list_trunc) and not any(list_done):
                    episode_timeout = True

                # Store transition in memory
                memory.store_transition(obs, global_state, actions, reward, 
                                        obs_, global_state_, done, values, log_probs)
                
                # Set new obs to current obs
                obs = obs_
                list_obs = list_obs_
                score += sum(list_reward)
                total_steps += 1
                episode_step += 1
                
            try:
                # When episode ends, get final values for proper advantage calculation
                global_state_ = obs_list_to_state_vector(list_obs_)
                _, last_values, _ = mappo_agents.choose_action(obs_, global_state_)
                
                # Finish episode and calculate advantages
                memory.finish_episode(last_values)
                
                # Learn from experience after episode is complete
                if not evaluate:
                    policy_losses, value_losses, entropy_losses = mappo_agents.learn(memory)
                    
                    # Log agent-specific training metrics to TensorBoard
                    for agent_idx in range(n_agents):
                        if agent_idx < len(policy_losses):
                            self.tb_logger.log_agent_metrics(
                                i,
                                agent_idx,
                                policy_losses[agent_idx],
                                value_losses[agent_idx],
                                entropy_losses[agent_idx]
                            )
            except Exception as e:
                self.get_logger().error(f'Error at episode end: {e}')
                # Print more detailed error information
                import traceback
                self.get_logger().error(traceback.format_exc())
                
            # Calculate the average score per robot
            episode_score = score / robot_number
            score_history.append(episode_score)
            step_history.append(episode_step)
            
            # Record episode outcome
            success_history.append(1 if episode_success else 0)
            collision_history.append(1 if episode_collision else 0)
            timeout_history.append(1 if episode_timeout else 0)
            
            # Get exploration coverage
            exploration_coverage = info.get('exploration_coverage', 0) if info else 0
            exploration_history.append(exploration_coverage)
            
            # Average the last 100 recent scores
            avg_score = np.mean(score_history[-100:]) if len(score_history) > 0 else 0
            
            # Log episode results to TensorBoard
            learning_rate = mappo_agents.get_lr()
            self.tb_logger.log_episode_results(
                i, episode_score, avg_score, episode_step,
                episode_success, episode_collision, episode_timeout,
                exploration_coverage, learning_rate
            )
            
            # Record metrics to CSV
            with open(metrics_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([i, episode_score, episode_step, 
                               int(episode_success), int(episode_collision), int(episode_timeout),
                               exploration_coverage, avg_score, learning_rate])
            
            # Generate training plots periodically
            if i % self.config.exp.plot_interval == 0 and i > 0:
                self.plot_training_metrics(score_history, step_history, success_history, 
                                          collision_history, timeout_history, exploration_history,
                                          training_metrics_dir, map_number, robot_number, timestamp)
            
            if not evaluate and len(score_history) >= 10:  # Wait for at least 10 episodes
                # Save when score improves
                if avg_score > best_score:
                    self.get_logger().info(f'New best score: {avg_score:.1f}! Saving checkpoint...')
                    mappo_agents.save_checkpoint()
                    best_score = avg_score
                
                # Also save periodically every N episodes
                if i % self.config.exp.save_interval == 0 and i > 0:
                    self.get_logger().info(f'Periodic save at episode {i}...')
                    # Save to a different directory to avoid overwriting best models
                    periodic_chkpt_dir = os.path.join(os.path.dirname(chkpt_dir_var), f'periodic_ep{i}')
                    # Log the absolute path where models will be saved
                    self.get_logger().info(f'Models will be saved to: {os.path.abspath(periodic_chkpt_dir)}')
                    # Ensure directory exists
                    os.makedirs(periodic_chkpt_dir, exist_ok=True)
                    
                    # Temporarily change the checkpoint directory
                    original_chkpt_dir = mappo_agents.agents[0].actor.chkpt_file
                    for agent in mappo_agents.agents:
                        agent.actor.chkpt_file = os.path.join(periodic_chkpt_dir, os.path.basename(agent.actor.chkpt_file))
                        agent.critic.chkpt_file = os.path.join(periodic_chkpt_dir, os.path.basename(agent.critic.chkpt_file))
                    
                    mappo_agents.save_checkpoint()
                    
                    # Restore original checkpoint directory
                    for agent_idx, agent in enumerate(mappo_agents.agents):
                        agent.actor.chkpt_file = original_chkpt_dir.replace('agent_0', f'agent_{agent_idx}')
                        agent.critic.chkpt_file = original_chkpt_dir.replace('agent_0_actor', f'agent_{agent_idx}_critic')
                    
            if i % PRINT_INTERVAL == 0:
                # Calculate success, collision, and timeout rates over last 100 episodes
                recent_success = np.mean(success_history[-100:]) * 100 if success_history else 0
                recent_collision = np.mean(collision_history[-100:]) * 100 if collision_history else 0
                recent_timeout = np.mean(timeout_history[-100:]) * 100 if timeout_history else 0
                
                self.get_logger().info('Episode: {}, Avg score: {:.1f}, Score: {:.1f}, Steps: {}, S/C/T: {:.0f}%/{:.0f}%/{:.0f}%, Expl: {:.1f}%'.format(
                    i, avg_score, episode_score, episode_step, recent_success, recent_collision, recent_timeout, 
                    exploration_coverage * 100))
                
    def plot_training_metrics(self, scores, steps, successes, collisions, timeouts, explorations, 
                              save_dir, map_number, robot_number, timestamp):
        """Generate and save plots of training metrics."""
        # Plot rolling average of rewards
        plt.figure(figsize=(10, 6))
        episodes = range(1, len(scores) + 1)
        plt.plot(episodes, scores, 'b-', alpha=0.3)
        
        # Add rolling average
        window_size = min(100, len(scores))
        if window_size > 0:
            rolling_mean = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size, len(scores) + 1), rolling_mean, 'r-')
        
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Training Scores')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'mappo_scores_map{map_number}_robots{robot_number}_{timestamp}.png'))
        plt.close()
        
        # Plot success/collision/timeout rates
        plt.figure(figsize=(10, 6))
        window_size = min(50, len(successes))  # Smaller window to show changes earlier
        
        # Calculate rates even for small datasets
        if len(successes) > 0:
            # Calculate raw rates for the entire history
            raw_success_rate = np.mean(successes) * 100
            raw_collision_rate = np.mean(collisions) * 100
            raw_timeout_rate = np.mean(timeouts) * 100
            
            # Plot individual data points with transparency
            plt.scatter(episodes, [s * 100 for s in successes], color='g', alpha=0.2, s=10)
            plt.scatter(episodes, [c * 100 for c in collisions], color='r', alpha=0.2, s=10)
            plt.scatter(episodes, [t * 100 for t in timeouts], color='y', alpha=0.2, s=10)
            
            # Add moving averages if we have enough data
            if window_size > 0:
                success_rate = np.convolve(successes, np.ones(window_size)/window_size, mode='valid') * 100
                collision_rate = np.convolve(collisions, np.ones(window_size)/window_size, mode='valid') * 100
                timeout_rate = np.convolve(timeouts, np.ones(window_size)/window_size, mode='valid') * 100
                
                plt.plot(range(window_size, len(successes) + 1), success_rate, 'g-', label=f'Success ({raw_success_rate:.1f}%)')
                plt.plot(range(window_size, len(collisions) + 1), collision_rate, 'r-', label=f'Collision ({raw_collision_rate:.1f}%)')
                plt.plot(range(window_size, len(timeouts) + 1), timeout_rate, 'y-', label=f'Timeout ({raw_timeout_rate:.1f}%)')
            else:
                # For very small datasets, just show the averages as horizontal lines
                plt.axhline(y=raw_success_rate, color='g', linestyle='-', label=f'Success ({raw_success_rate:.1f}%)')
                plt.axhline(y=raw_collision_rate, color='r', linestyle='-', label=f'Collision ({raw_collision_rate:.1f}%)')
                plt.axhline(y=raw_timeout_rate, color='y', linestyle='-', label=f'Timeout ({raw_timeout_rate:.1f}%)')
        
        plt.xlabel('Episode')
        plt.ylabel('Rate (%)')
        plt.title('Episode Outcomes')
        plt.legend()
        plt.grid(True)
        plt.ylim([-5, 105])  # Set y-axis limits for percentage
        plt.savefig(os.path.join(save_dir, f'mappo_outcomes_map{map_number}_robots{robot_number}_{timestamp}.png'))
        plt.close()
        
        # Plot exploration coverage
        plt.figure(figsize=(10, 6))
        
        # Convert exploration values to percentages for better visibility
        exploration_percentages = [e * 100 for e in explorations]
        
        # Plot individual data points with slight transparency
        plt.scatter(episodes, exploration_percentages, color='g', alpha=0.4, s=15)
        
        # Add rolling average if we have enough data
        window_size = min(50, len(explorations))
        if window_size > 0 and any(e > 0 for e in explorations):
            rolling_mean = np.convolve(exploration_percentages, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size, len(explorations) + 1), rolling_mean, 'b-', linewidth=2)
        
        # Calculate max exploration so far
        max_exploration = max(exploration_percentages) if exploration_percentages else 0
        avg_exploration = sum(exploration_percentages) / len(exploration_percentages) if exploration_percentages else 0
        
        # Add a horizontal line for the average exploration
        if exploration_percentages:
            plt.axhline(y=avg_exploration, color='r', linestyle='--', 
                        label=f'Avg: {avg_exploration:.2f}%, Max: {max_exploration:.2f}%')
        
        plt.xlabel('Episode')
        plt.ylabel('Exploration Coverage (%)')
        plt.title('Exploration Efficiency')
        plt.grid(True)
        
        # Set appropriate y-axis limits - always show at least 0-10% range
        # for small values, but expand if we have larger values
        y_max = max(10, max_exploration * 1.2) if max_exploration > 0 else 10
        plt.ylim([-0.5, y_max])
        
        if exploration_percentages:
            plt.legend()
            
        plt.savefig(os.path.join(save_dir, f'mappo_exploration_map{map_number}_robots{robot_number}_{timestamp}.png'))
        plt.close()


def main(args=None):
    rclpy.init(args=args)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MAPPO training for multi-robot exploration')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--map_number', type=int, help='Map number to use')
    parser.add_argument('--robot_number', type=int, help='Number of robots to use')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    
    # Parse known args to handle ROS 2 arguments properly
    parsed_args, _ = parser.parse_known_args(args=args)
    
    # Set environment variables if specified
    if parsed_args.map_number:
        os.environ['map_number'] = str(parsed_args.map_number)
    if parsed_args.robot_number:
        os.environ['robot_number'] = str(parsed_args.robot_number)
        
    # Create node with config path
    node = MAPPONode(config_path=parsed_args.config)
    
    # Cleanup
    node.tb_logger.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
