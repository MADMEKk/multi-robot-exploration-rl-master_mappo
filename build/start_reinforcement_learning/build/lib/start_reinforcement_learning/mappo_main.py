import os
import rclpy
from rclpy.node import Node
import numpy as np

from start_reinforcement_learning.logic import Env
from start_reinforcement_learning.mappo_algorithm.mappo import MAPPO
from start_reinforcement_learning.mappo_algorithm.buffer import MultiAgentReplayBuffer
import torch as T
import gc
from ament_index_python.packages import get_package_share_directory

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
        N_GAMES = 5000
        total_steps = 0
        score_history = []
        evaluate = False
        best_score = 0

        # Test network
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
            score_history.append(score / robot_number)
            # Average the last 100 recent scores
            avg_score = np.mean(score_history[-100:]) if len(score_history) > 0 else 0
            
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
                self.get_logger().info('Episode: {}, Average score: {:.1f}, Episode Score: {:.1f}'.format(
                    i, avg_score, score / robot_number))


def main(args=None):
    rclpy.init(args=args)
    
    map_number = int(os.getenv('map_number', '1'))
    robot_number = int(os.getenv('robot_number', '3'))
    node = MAPPONode(map_number, robot_number)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
