import os
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from datetime import datetime
import torch as T
from start_reinforcement_learning.metrics_analyzer import MetricsAnalyzer

from start_reinforcement_learning.logic import Env
from start_reinforcement_learning.mappo_algorithm.mappo import MAPPO
from start_reinforcement_learning.maddpg_algorithm.maddpg import MADDPG
from start_reinforcement_learning.mappo_algorithm.buffer import MultiAgentReplayBuffer as MAPPOBuffer
from start_reinforcement_learning.maddpg_algorithm.buffer import MultiAgentReplayBuffer as MADDPGBuffer
from ament_index_python.packages import get_package_share_directory

# Convert list of arrays to one flat array of observations
def obs_list_to_state_vector(observation):
    state = np.array([])    
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

class AlgorithmComparison(Node):
    def __init__(self, map_number, robot_number, n_episodes=100, save_dir='comparison_results'):
        super().__init__('algorithm_comparison_node')
        
        # Access the parameters passed from the launch file
        self.map_number = self.declare_parameter('map_number', map_number).get_parameter_value().integer_value
        self.robot_number = self.declare_parameter('robot_number', robot_number).get_parameter_value().integer_value
        self.n_episodes = self.declare_parameter('n_episodes', n_episodes).get_parameter_value().integer_value
        self.save_dir = save_dir
        
        self.get_logger().info(f"Starting algorithm comparison with map={self.map_number}, robots={self.robot_number}, episodes={self.n_episodes}")
        
        # Create directory for saving results if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize environment
        self.env = Env(self.robot_number, self.map_number)
        self.n_agents = self.env.number_of_robots
        
        # Get observation and action dimensions
        self.actor_dims = self.env.observation_space()
        self.critic_dims = sum(self.actor_dims)
        self.n_actions = 9  # 3 linear velocity options x 3 angular velocity options
        
        # Initialize metrics tracking
        self.metrics = {
            'mappo': {
                'episode_rewards': [],
                'episode_steps': [],
                'goals_reached': 0,
                'collisions': 0,
                'execution_times': []
            },
            'maddpg': {
                'episode_rewards': [],
                'episode_steps': [],
                'goals_reached': 0,
                'collisions': 0,
                'execution_times': []
            }
        }
        
        # Initialize algorithms
        self._init_algorithms()
        
    def _init_algorithms(self):
        # MAPPO checkpoint directory
        mappo_chkpt_dir = os.path.join(
            get_package_share_directory('start_reinforcement_learning'),
            'start_reinforcement_learning', 'deep_learning_weights', 'mappo'
        )
        
        # MADDPG checkpoint directory
        maddpg_chkpt_dir = os.path.join(
            get_package_share_directory('start_reinforcement_learning'),
            'start_reinforcement_learning', 'deep_learning_weights', 'maddpg'
        )
        
        # Initialize MAPPO
        self.mappo_agents = MAPPO(
            self.actor_dims, self.critic_dims, self.n_agents, self.n_actions, 
            fc1=512, fc2=512, tau=0.00025,
            alpha=3e-4, beta=1e-3, scenario='robot',
            chkpt_dir=mappo_chkpt_dir, node_logger=self
        )
        
        # Initialize MADDPG
        self.maddpg_agents = MADDPG(
            self.actor_dims, self.critic_dims, self.n_agents, self.n_actions, 
            fc1=512, fc2=512, tau=0.00025,
            alpha=1e-4, beta=1e-3, scenario='robot',
            chkpt_dir=maddpg_chkpt_dir, node_logger=self
        )
        
        # Load pre-trained models
        self.mappo_agents.load_checkpoint()
        self.maddpg_agents.load_checkpoint()
        
    def evaluate_algorithm(self, algorithm_name, max_steps=500):
        """
        Evaluate a single algorithm for n_episodes
        """
        if algorithm_name == 'mappo':
            agent = self.mappo_agents
        elif algorithm_name == 'maddpg':
            agent = self.maddpg_agents
        else:
            self.get_logger().error(f"Unknown algorithm: {algorithm_name}")
            return
            
        self.get_logger().info(f"Evaluating {algorithm_name.upper()} for {self.n_episodes} episodes")
        
        for episode in range(self.n_episodes):
            # Reset environment
            obs = self.env.reset()
            list_obs = list(obs.values())
            
            episode_reward = 0
            episode_steps = 0
            done = [False] * self.n_agents
            terminal = [False] * self.n_agents
            
            # Track if any robot reached goal or collided
            goal_reached = False
            collision_occurred = False
            
            # Start timing
            start_time = time.time()
            
            # Run episode
            while not any(terminal) and episode_steps < max_steps:
                # Get global state
                global_state = obs_list_to_state_vector(list_obs)
                
                # Choose action based on algorithm
                if algorithm_name == 'mappo':
                    actions, _, _ = agent.choose_action(obs, global_state, deterministic=True)
                else:  # maddpg
                    actions = agent.choose_action(obs)
                
                # Take action in environment
                obs_, reward, done, truncated, info = self.env.step(actions)
                
                # Convert to lists
                list_done = list(done.values())
                list_reward = list(reward.values())
                list_obs_ = list(obs_.values())
                list_trunc = list(truncated.values())
                
                # Check if episode is done
                terminal = [d or t for d, t in zip(list_done, list_trunc)]
                
                # Check for goal or collision
                for d, r in zip(list_done, list_reward):
                    if d and r > 0:  # Positive reward on done means goal reached
                        goal_reached = True
                    elif d and r < 0:  # Negative reward on done means collision
                        collision_occurred = True
                
                # Update state
                obs = obs_
                list_obs = list_obs_
                episode_reward += sum(list_reward)
                episode_steps += 1
            
            # End timing
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Store metrics
            self.metrics[algorithm_name]['episode_rewards'].append(episode_reward)
            self.metrics[algorithm_name]['episode_steps'].append(episode_steps)
            self.metrics[algorithm_name]['execution_times'].append(execution_time)
            
            if goal_reached:
                self.metrics[algorithm_name]['goals_reached'] += 1
            
            if collision_occurred:
                self.metrics[algorithm_name]['collisions'] += 1
            
            # Log progress
            if (episode + 1) % 10 == 0:
                self.get_logger().info(
                    f"{algorithm_name.upper()} - Episode: {episode+1}/{self.n_episodes}, "
                    f"Reward: {episode_reward:.2f}, Steps: {episode_steps}, "
                    f"Time: {execution_time:.2f}s"
                )
    
    def run_comparison(self):
        """
        Run comparison between MAPPO and MADDPG
        """
        # Evaluate MAPPO
        self.evaluate_algorithm('mappo')
        
        # Evaluate MADDPG
        self.evaluate_algorithm('maddpg')
        
        # Save results
        self.save_results()
        
        # Plot comparison
        self.plot_comparison()
        
    def save_results(self):
        """
        Save metrics to CSV files and generate comprehensive analysis
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save episode rewards
        with open(f"{self.save_dir}/rewards_{timestamp}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'MAPPO', 'MADDPG'])
            for i in range(self.n_episodes):
                writer.writerow([
                    i+1, 
                    self.metrics['mappo']['episode_rewards'][i],
                    self.metrics['maddpg']['episode_rewards'][i]
                ])
        
        # Save episode steps
        with open(f"{self.save_dir}/steps_{timestamp}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'MAPPO', 'MADDPG'])
            for i in range(self.n_episodes):
                writer.writerow([
                    i+1, 
                    self.metrics['mappo']['episode_steps'][i],
                    self.metrics['maddpg']['episode_steps'][i]
                ])
        
        # Save execution times
        with open(f"{self.save_dir}/times_{timestamp}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'MAPPO', 'MADDPG'])
            for i in range(self.n_episodes):
                writer.writerow([
                    i+1, 
                    self.metrics['mappo']['execution_times'][i],
                    self.metrics['maddpg']['execution_times'][i]
                ])
        
        # Save summary statistics
        with open(f"{self.save_dir}/summary_{timestamp}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'MAPPO', 'MADDPG'])
            writer.writerow([
                'Avg Reward', 
                np.mean(self.metrics['mappo']['episode_rewards']),
                np.mean(self.metrics['maddpg']['episode_rewards'])
            ])
            writer.writerow([
                'Avg Steps', 
                np.mean(self.metrics['mappo']['episode_steps']),
                np.mean(self.metrics['maddpg']['episode_steps'])
            ])
            writer.writerow([
                'Goals Reached', 
                self.metrics['mappo']['goals_reached'],
                self.metrics['maddpg']['goals_reached']
            ])
            writer.writerow([
                'Collisions', 
                self.metrics['mappo']['collisions'],
                self.metrics['maddpg']['collisions']
            ])
            writer.writerow([
                'Avg Execution Time (s)', 
                np.mean(self.metrics['mappo']['execution_times']),
                np.mean(self.metrics['maddpg']['execution_times'])
            ])
        
        # Generate comprehensive analysis using the metrics analyzer
        analyzer = MetricsAnalyzer(self.save_dir)
        report_path = analyzer.generate_comprehensive_report(self.metrics)
        
        self.get_logger().info(f"Results saved to {self.save_dir}")
        self.get_logger().info(f"Comprehensive report generated at {report_path}")
    
    def plot_comparison(self):
        """
        Generate basic comparison plots (detailed plots are handled by MetricsAnalyzer)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot rewards
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['mappo']['episode_rewards'], label='MAPPO')
        plt.plot(self.metrics['maddpg']['episode_rewards'], label='MADDPG')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.legend()
        
        # Plot steps
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['mappo']['episode_steps'], label='MAPPO')
        plt.plot(self.metrics['maddpg']['episode_steps'], label='MADDPG')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Steps')
        plt.legend()
        
        # Plot execution times
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['mappo']['execution_times'], label='MAPPO')
        plt.plot(self.metrics['maddpg']['execution_times'], label='MADDPG')
        plt.xlabel('Episode')
        plt.ylabel('Time (s)')
        plt.title('Execution Time')
        plt.legend()
        
        # Plot success rates
        plt.subplot(2, 2, 4)
        labels = ['MAPPO', 'MADDPG']
        goals = [self.metrics['mappo']['goals_reached'], self.metrics['maddpg']['goals_reached']]
        collisions = [self.metrics['mappo']['collisions'], self.metrics['maddpg']['collisions']]
        timeouts = [self.n_episodes - g - c for g, c in zip(goals, collisions)]
        
        x = np.arange(len(labels))
        width = 0.25
        
        plt.bar(x - width, goals, width, label='Goals Reached')
        plt.bar(x, collisions, width, label='Collisions')
        plt.bar(x + width, timeouts, width, label='Timeouts')
        
        plt.xlabel('Algorithm')
        plt.ylabel('Count')
        plt.title('Episode Outcomes')
        plt.xticks(x, labels)
        plt.legend()
        
        # Save plots
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/comparison_plots_{timestamp}.png")
        plt.close()
        
        self.get_logger().info(f"Plots saved to {self.save_dir}/comparison_plots_{timestamp}.png")

def main(args=None):
    rclpy.init(args=args)
    
    map_number = int(os.getenv('map_number', '1'))
    robot_number = int(os.getenv('robot_number', '2'))
    n_episodes = int(os.getenv('n_episodes', '100'))
    
    comparison = AlgorithmComparison(map_number, robot_number, n_episodes)
    comparison.run_comparison()
    
    comparison.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
