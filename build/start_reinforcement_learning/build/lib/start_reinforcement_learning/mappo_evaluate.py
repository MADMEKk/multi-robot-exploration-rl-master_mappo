import os
import rclpy
from rclpy.node import Node
import numpy as np

from start_reinforcement_learning.logic import Env
from start_reinforcement_learning.mappo_algorithm.mappo import MAPPO
import torch as T
from ament_index_python.packages import get_package_share_directory

# Convert list of arrays to one flat array of observations
def obs_list_to_state_vector(observation):
    state = np.array([])    
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

# Main function that runs the MAPPO algorithm in evaluation mode
class MAPPOEvaluateNode(Node):
    def __init__(self, map_number, robot_number):
        super().__init__('mappo_evaluate_node')

        # Access the parameters passed from the launch file
        map_number = self.declare_parameter('map_number', 1).get_parameter_value().integer_value
        robot_number = self.declare_parameter('robot_number', 3).get_parameter_value().integer_value
        model_episode = self.declare_parameter('model_episode', 0).get_parameter_value().integer_value

        self.get_logger().info(f"Evaluation Mode - Map number: {map_number}")
        self.get_logger().info(f"Evaluation Mode - Robot number: {robot_number}")

        # Set environment with action size
        env = Env(robot_number, map_number)
        n_agents = env.number_of_robots
        
        actor_dims = env.observation_space()
        critic_dims = sum(actor_dims)

        # Action space is discrete, one of 9 actions (3 linear x 3 angular)
        n_actions = 9  # 3 linear velocity options x 3 angular velocity options

        # Use direct path instead of get_package_share_directory which might fail
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Check if a specific model path was provided via environment variable
        model_path_env = os.getenv('model_path')
        
        if model_path_env and os.path.exists(model_path_env):
            # Use the specific model path from environment variable
            chkpt_dir_var = model_path_env
            self.get_logger().info(f"Using model path from environment variable: {chkpt_dir_var}")
        else:
            # Use the traditional method with model_episode
            # Base checkpoint directory
            chkpt_dir_var = os.path.join(base_path, 'start_reinforcement_learning', 'deep_learning_weights', 'mappo')
            
            # If using src directory instead of build
            src_chkpt_dir = os.path.join('/home/aladine/memoir/multi-robot-exploration-rl-master/src', 
                                        'start_reinforcement_learning', 'start_reinforcement_learning', 
                                        'deep_learning_weights', 'mappo')
            
            # Check if src directory exists and use it if it does
            if os.path.exists(src_chkpt_dir):
                chkpt_dir_var = src_chkpt_dir
                
            # Find best model directory if available
            best_models_dir = os.path.join(chkpt_dir_var, f'map{map_number}_robots{robot_number}_best')
            
            # First, try to load from the best models directory if model_episode is 0
            if model_episode == 0 and os.path.exists(best_models_dir):
                # Check if a model tracker file exists
                tracker_file = os.path.join(os.path.dirname(best_models_dir), 
                                         f'model_tracker_map{map_number}_robots{robot_number}.json')
                
                if os.path.exists(tracker_file):
                    # Load the best model from the tracker file
                    import json
                    with open(tracker_file, 'r') as f:
                        tracker_data = json.load(f)
                        
                    best_model_key = tracker_data.get('best_model')
                    if best_model_key and best_model_key in tracker_data.get('models', {}):
                        best_model_path = tracker_data['models'][best_model_key].get('path')
                        if best_model_path and os.path.exists(best_model_path):
                            chkpt_dir_var = best_model_path
                            self.get_logger().info(f"Using best model from tracker: {best_model_key}")
                            self.get_logger().info(f"Best model score: {tracker_data['models'][best_model_key].get('score')}")
                            self.get_logger().info(f"Goal success rate: {tracker_data['models'][best_model_key].get('goal_success_rate')}")
            
            # If no best model found or model_episode > 0, use the periodic directory
            if model_episode > 0:
                chkpt_dir_var = os.path.join(os.path.dirname(chkpt_dir_var), f'periodic_ep{model_episode}')
            # If model_episode is 0 and no best model was found, use the default directory
            elif model_episode == 0 and not ('best_model_key' in locals() and best_model_key):
                chkpt_dir_var = os.path.join(chkpt_dir_var, f'map{map_number}_robots{robot_number}')
        
        self.get_logger().info(f"Loading model from: {chkpt_dir_var}")
        
        # Initialize main algorithm
        mappo_agents = MAPPO(actor_dims, critic_dims, n_agents, n_actions, 
                             fc1=512, fc2=512, tau=0.00025,
                             alpha=3e-4, beta=1e-3, scenario='robot',
                             chkpt_dir=chkpt_dir_var, node_logger=self)

        # Load the trained model
        mappo_agents.load_checkpoint()
        self.get_logger().info("Loaded trained model for evaluation")

        PRINT_INTERVAL = 1
        N_GAMES = 100  # Number of evaluation episodes
        score_history = []

        # Evaluation loop
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
                # Use deterministic=True for evaluation
                actions, _, _ = mappo_agents.choose_action(obs, global_state, deterministic=True)
                
                # Use step function to get next state and reward info as well as if the episode is 'done'
                obs_, reward, done, truncated, info = env.step(actions)
                
                # Convert dict -> list of arrays
                list_done = list(done.values())
                list_reward = list(reward.values())
                list_obs_ = list(obs_.values())
                list_trunc = list(truncated.values())
                
                # Check if episode is done
                terminal = [d or t for d, t in zip(list_done, list_trunc)]

                # Set new obs to current obs
                obs = obs_
                list_obs = list_obs_
                score += sum(list_reward)
                episode_step += 1
                
            # Calculate the average score per robot
            score_history.append(score / robot_number)
            # Average the scores
            avg_score = np.mean(score_history)
                    
            if i % PRINT_INTERVAL == 0:
                self.get_logger().info('Evaluation Episode: {}, Average score: {:.1f}, Episode Score: {:.1f}, Steps: {}'.format(
                    i, avg_score, score / robot_number, episode_step))

        # Final evaluation results
        self.get_logger().info(f"Evaluation complete. Final average score over {N_GAMES} episodes: {avg_score:.2f}")


def main(args=None):
    rclpy.init(args=args)
    
    map_number = int(os.getenv('map_number', '1'))
    robot_number = int(os.getenv('robot_number', '3'))
    model_episode = int(os.getenv('model_episode', '0'))  # 0 means use best model, otherwise use periodic save
    
    try:
        node = MAPPOEvaluateNode(map_number, robot_number)
    except Exception as e:
        print(f"Error initializing evaluation node: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
