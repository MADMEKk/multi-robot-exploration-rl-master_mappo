import os
import rclpy
from rclpy.node import Node
import numpy as np
import csv
from datetime import datetime
import time

from start_reinforcement_learning.logic import Env
from start_reinforcement_learning.mappo_algorithm.mappo import MAPPO
from start_reinforcement_learning.mappo_algorithm.buffer import MultiAgentReplayBuffer
import torch as T
import gc
from ament_index_python.packages import get_package_share_directory
import matplotlib.pyplot as plt

# Convert list of arrays to one flat array of observations
def obs_list_to_state_vector(observation):
    state = np.array([])    
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

# Main function that runs the MAPPO algorithm
class MAPPONode(Node):
    def __init__(self, map_number, robot_number):
        super().__init__('mappo_node')

        # Access the parameters passed from the launch file
        map_number = self.declare_parameter('map_number', 1).get_parameter_value().integer_value
        robot_number = self.declare_parameter('robot_number', 3).get_parameter_value().integer_value

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
        chkpt_dir_var = os.path.join(base_path, 'start_reinforcement_learning', 'deep_learning_weights', 'mappo', f'map{map_number}_robots{robot_number}')
        self.get_logger().info(f"Checkpoint directory: {chkpt_dir_var}")
        
        # Create training metrics directory
        training_metrics_dir = os.path.join(base_path, 'start_reinforcement_learning', 'training_metrics')
        os.makedirs(training_metrics_dir, exist_ok=True)
        
        # Timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize main algorithm
        mappo_agents = MAPPO(actor_dims, critic_dims, n_agents, n_actions, 
                             fc1=512, fc2=512, tau=0.00025,
                             alpha=3e-4, beta=1e-3, scenario='robot',
                             chkpt_dir=chkpt_dir_var, node_logger=self)

        # Initialize memory
        memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                                        n_actions, n_agents, batch_size=2048,
                                        gamma=0.99, gae_lambda=0.95)

        PRINT_INTERVAL = 10
        N_GAMES = 10000
        total_steps = 0
        score_history = []
        step_history = []
        success_history = []
        collision_history = []
        timeout_history = []
        exploration_history = []
        best_score = -20

        # Create training metrics file
        metrics_filename = os.path.join(training_metrics_dir, f"mappo_training_map{map_number}_robots{robot_number}_{timestamp}.csv")
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
                memory.store_transition(list_obs, global_state, actions, list_reward, 
                                       list_obs_, global_state_, list_done, values, log_probs)
                
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
                
                # Convert last_values to a format that the buffer can handle
                if isinstance(last_values, dict):
                    list_last_values = list(last_values.values())
                else:
                    # If it's not a dict, create a list with one value per agent
                    list_last_values = [0.0] * n_agents
                
                # Ensure list_last_values is not empty
                if not list_last_values or len(list_last_values) == 0:
                    list_last_values = [0.0] * n_agents
                
                # Convert any None values to 0.0
                list_last_values = [0.0 if v is None else v for v in list_last_values]
                
                # Make sure we have one value per agent
                if len(list_last_values) < n_agents:
                    # Pad with zeros if needed
                    list_last_values.extend([0.0] * (n_agents - len(list_last_values)))
                elif len(list_last_values) > n_agents:
                    # Truncate if needed
                    list_last_values = list_last_values[:n_agents]
                
                self.get_logger().info(f"Last values shape: {len(list_last_values)}")
                
                # Finish episode and calculate advantages
                memory.finish_episode(list_last_values)
                
                # Learn from experience after episode is complete
                if not evaluate:
                    mappo_agents.learn(memory)
            except Exception as e:
                self.get_logger().error(f'Error at episode end: {e}')
                # Print more detailed error information
                import traceback
                self.get_logger().error(traceback.format_exc())
                
                # Reset memory for the next episode if there was an error
                memory.current_episode = {
                    'states': [],
                    'next_states': [],
                    'actions': [],
                    'rewards': [],
                    'values': [],
                    'log_probs': [],
                    'dones': [],
                    'individual_obs': [[] for _ in range(n_agents)],
                    'individual_next_obs': [[] for _ in range(n_agents)]
                }
                memory.episode_step = 0
                
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
            
            # Record metrics to CSV
            with open(metrics_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([i, episode_score, episode_step, 
                               int(episode_success), int(episode_collision), int(episode_timeout),
                               exploration_coverage, avg_score, mappo_agents.get_lr()])
            
            # Generate training plots periodically
            if i % 50 == 0 and i > 0:
                self.plot_training_metrics(score_history, step_history, success_history, 
                                          collision_history, timeout_history, exploration_history,
                                          training_metrics_dir, map_number, robot_number, timestamp)
            
            if not evaluate and len(score_history) >= 10:  # Wait for at least 10 episodes
                # Save when score improves
                if avg_score > best_score:
                    self.get_logger().info(f'New best score: {avg_score:.1f}! Saving checkpoint...')
                    mappo_agents.save_checkpoint()
                    best_score = avg_score
                
                # Also save periodically every 50 episodes
                if i % 50 == 0 and i > 0:
                    self.get_logger().info(f'Periodic save at episode {i}...')
                    # Save to a different directory to avoid overwriting best models
                    periodic_chkpt_dir = os.path.join(os.path.dirname(chkpt_dir_var), f'periodic_ep{i}')
                    # Log the absolute path where models will be saved
                    self.get_logger().info(f'Models will be saved to: {os.path.abspath(periodic_chkpt_dir)}')
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
        window_size = min(100, len(successes))
        if window_size > 0:
            success_rate = np.convolve(successes, np.ones(window_size)/window_size, mode='valid') * 100
            collision_rate = np.convolve(collisions, np.ones(window_size)/window_size, mode='valid') * 100
            timeout_rate = np.convolve(timeouts, np.ones(window_size)/window_size, mode='valid') * 100
            
            plt.plot(range(window_size, len(successes) + 1), success_rate, 'g-', label='Success')
            plt.plot(range(window_size, len(collisions) + 1), collision_rate, 'r-', label='Collision')
            plt.plot(range(window_size, len(timeouts) + 1), timeout_rate, 'y-', label='Timeout')
            
        plt.xlabel('Episode')
        plt.ylabel('Rate (%)')
        plt.title('Episode Outcomes')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'mappo_outcomes_map{map_number}_robots{robot_number}_{timestamp}.png'))
        plt.close()
        
        # Plot exploration coverage
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, explorations, 'g-', alpha=0.3)
        
        # Add rolling average
        window_size = min(100, len(explorations))
        if window_size > 0:
            rolling_mean = np.convolve(explorations, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size, len(explorations) + 1), rolling_mean, 'b-')
        
        plt.xlabel('Episode')
        plt.ylabel('Exploration Coverage')
        plt.title('Exploration Efficiency')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'mappo_exploration_map{map_number}_robots{robot_number}_{timestamp}.png'))
        plt.close()


def main(args=None):
    rclpy.init(args=args)
    
    map_number = int(os.getenv('map_number', '1'))
    robot_number = int(os.getenv('robot_number', '3'))
    node = MAPPONode(map_number, robot_number)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
