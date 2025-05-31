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

        chkpt_dir_var = os.path.join(get_package_share_directory('start_reinforcement_learning'),
                                    'start_reinforcement_learning', 'deep_learning_weights', 'mappo')
        
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
                
            # When episode ends, get final values for proper advantage calculation
            global_state_ = obs_list_to_state_vector(list_obs_)
            _, last_values, _ = mappo_agents.choose_action(obs_, global_state_)
            list_last_values = list(last_values.values())
            
            # Finish episode and calculate advantages
            memory.finish_episode(list_last_values)
            
            # Learn from experience after episode is complete
            if not evaluate:
                mappo_agents.learn(memory)
                
            # Calculate the average score per robot
            score_history.append(score / robot_number)
            # Average the last 100 recent scores
            avg_score = np.mean(score_history[-100:]) if len(score_history) > 0 else 0
            
            if not evaluate and len(score_history) >= 10:  # Wait for at least 10 episodes
                if avg_score > best_score:
                    mappo_agents.save_checkpoint()
                    best_score = avg_score
                    
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
