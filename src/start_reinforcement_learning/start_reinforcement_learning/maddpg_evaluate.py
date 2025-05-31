import os
import rclpy
from rclpy.node import Node
import numpy as np

from start_reinforcement_learning.env_logic.logic import Env
from start_reinforcement_learning.maddpg_algorithm.maddpg import MADDPG
import torch as T
from ament_index_python.packages import get_package_share_directory

# Convert list of arrays to one flat array of observations
def obs_list_to_state_vector(observation):
    state = np.array([])    
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

# Main function that runs the MADDPG evaluation
class MADDPGEvaluateNode(Node):
    def __init__(self, map_number, robot_number, model_episode):
        super().__init__('maddpg_evaluate_node')

        # Access the parameters passed from the launch file
        map_number = self.declare_parameter('map_number', 1).get_parameter_value().integer_value
        robot_number = self.declare_parameter('robot_number', 3).get_parameter_value().integer_value
        model_episode = self.declare_parameter('model_episode', 0).get_parameter_value().integer_value

        self.get_logger().info(f"Evaluating MADDPG with:")
        self.get_logger().info(f"Map number: {map_number}")
        self.get_logger().info(f"Robot number: {robot_number}")
        self.get_logger().info(f"Model episode: {model_episode}")

        # Set environment with action size
        env = Env(robot_number, map_number)
        n_agents = env.number_of_robots
        
        actor_dims = env.observation_space()
        critic_dims = sum(actor_dims)

        # Action space is discrete, one of 4 actions, look in env
        n_actions = env.action_space()

        # Use direct path instead of get_package_share_directory which might fail
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Determine which model to load based on model_episode parameter
        if model_episode > 0:
            # Load a specific periodic save
            chkpt_dir_var = os.path.join(base_path, 'start_reinforcement_learning', 'deep_learning_weights', 
                                        'maddpg', f'periodic_ep{model_episode}')
            self.get_logger().info(f"Loading periodic save from episode {model_episode}")
        else:
            # Load the best model
            chkpt_dir_var = os.path.join(base_path, 'start_reinforcement_learning', 'deep_learning_weights', 
                                        'maddpg', f'map{map_number}_robots{robot_number}')
            self.get_logger().info(f"Loading best model")
            
        self.get_logger().info(f"Checkpoint directory: {chkpt_dir_var}")
        
        # Check if the directory exists
        if not os.path.exists(chkpt_dir_var):
            # Try looking in the src directory instead
            src_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
            if model_episode > 0:
                chkpt_dir_var = os.path.join(src_base_path, 'start_reinforcement_learning', 'start_reinforcement_learning', 
                                            'deep_learning_weights', 'maddpg', f'periodic_ep{model_episode}')
            else:
                chkpt_dir_var = os.path.join(src_base_path, 'start_reinforcement_learning', 'start_reinforcement_learning', 
                                            'deep_learning_weights', 'maddpg', f'map{map_number}_robots{robot_number}')
            self.get_logger().info(f"Trying alternative checkpoint directory: {chkpt_dir_var}")
        
        # Initialize main algorithm
        maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                               fc1=512, fc2=512, tau=0.00025,
                               alpha=1e-4, beta=1e-3, scenario='robot',
                               chkpt_dir=chkpt_dir_var, node_logger=self)

        # Load the trained model
        try:
            maddpg_agents.load_checkpoint()
            self.get_logger().info("Successfully loaded checkpoint")
        except Exception as e:
            self.get_logger().error(f"Failed to load checkpoint: {str(e)}")
            return

        PRINT_INTERVAL = 10
        N_GAMES = 100  # Run 100 evaluation episodes
        score_history = []

        # Run evaluation episodes
        for i in range(N_GAMES):
            # reset to get initial observation
            obs = env.reset()
            # Convert dict -> list of arrays
            list_obs = list(obs.values())
            score = 0
            done = [False]*n_agents
            terminal = [False] * n_agents
            episode_step = 0

            # Run until episode terminates
            while not any(terminal):
                # Get the actions that the algorithm thinks are best in given observation
                # Use deterministic=True for evaluation (no exploration noise)
                actions = maddpg_agents.choose_action(obs, evaluate=True)
                
                # use step function to get next state and reward info
                obs_, reward, done, truncated, info = env.step(actions)
                
                # Convert dict -> list of arrays
                list_done = list(done.values())
                list_reward = list(reward.values())
                list_trunc = list(truncated.values())
                
                terminal = [d or t for d, t in zip(list_done, list_trunc)]

                # Set new obs to current obs
                obs = obs_
                score += sum(list_reward)
                episode_step += 1
                
            # Calculate the average score per robot
            episode_score = score/robot_number
            score_history.append(episode_score)
            
            if i % PRINT_INTERVAL == 0 or i == N_GAMES - 1:
                avg_score = np.mean(score_history[-PRINT_INTERVAL:])
                self.get_logger().info(f'Episode: {i}, Score: {episode_score:.1f}, Average score: {avg_score:.1f}')

        # Final evaluation results
        overall_avg_score = np.mean(score_history)
        self.get_logger().info(f'Evaluation complete. Overall average score: {overall_avg_score:.1f}')


def main(args=None):
    rclpy.init(args=args)
    
    map_number = int(os.getenv('map_number', '1'))
    robot_number = int(os.getenv('robot_number', '3'))
    model_episode = int(os.getenv('model_episode', '0'))
    
    node = MADDPGEvaluateNode(map_number, robot_number, model_episode)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
    return 0


if __name__ == '__main__':
    main()
