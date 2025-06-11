import os
import rclpy
from rclpy.node import Node
import numpy as np
import csv
import time
from datetime import datetime

from start_reinforcement_learning.logic import Env
from start_reinforcement_learning.mappo_algorithm.mappo import MAPPO
from start_reinforcement_learning.evaluation_metrics import EvaluationMetrics
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
        
        # Add parameters for goal position
        goal_x = self.declare_parameter('goal_x', -999.0).get_parameter_value().double_value
        goal_y = self.declare_parameter('goal_y', -999.0).get_parameter_value().double_value

        self.get_logger().info(f"Evaluation Mode - Map number: {map_number}")
        self.get_logger().info(f"Evaluation Mode - Robot number: {robot_number}")

        # Set environment with action size
        env = Env(robot_number, map_number)
        n_agents = env.number_of_robots
        
        # Set custom goal position if provided
        if goal_x > -999.0 and goal_y > -999.0:
            self.get_logger().info(f"Using custom goal position: ({goal_x}, {goal_y})")
            env.set_goal(goal_x, goal_y)
        else:
            self.get_logger().info("Using default goal position from environment")
        
        actor_dims = env.observation_space()
        critic_dims = sum(actor_dims)

        # Action space is discrete, one of 9 actions (3 linear x 3 angular)
        n_actions = 9  # 3 linear velocity options x 3 angular velocity options

        # Use direct path instead of get_package_share_directory which might fail
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Base checkpoint directory
        chkpt_dir_var = os.path.join(base_path, 'start_reinforcement_learning', 'deep_learning_weights', 'mappo')
        
        # If using src directory instead of build
        src_chkpt_dir = os.path.join('/home/aladine/memoir/multi-robot-exploration-rl-master/src', 
                                    'start_reinforcement_learning', 'start_reinforcement_learning', 
                                    'deep_learning_weights', 'mappo')
        
        # Check if src directory exists and use it if it does
        if os.path.exists(src_chkpt_dir):
            chkpt_dir_var = src_chkpt_dir
            
        # Add map and robot numbers to path
        chkpt_dir_var = os.path.join(chkpt_dir_var, f'map{map_number}_robots{robot_number}')
        
        # If model_episode is specified, use the periodic save directory
        if model_episode > 0:
            chkpt_dir_var = os.path.join(os.path.dirname(chkpt_dir_var), f'periodic_ep{model_episode}')
            
        self.get_logger().info(f"Loading model from: {chkpt_dir_var}")
        
        # Initialize main algorithm
        mappo_agents = MAPPO(actor_dims, critic_dims, n_agents, n_actions, 
                             fc1=512, fc2=512, tau=0.00025,
                             alpha=3e-4, beta=1e-3, scenario='robot',
                             chkpt_dir=chkpt_dir_var, node_logger=self)

        # Load the trained model
        mappo_agents.load_checkpoint()
        self.get_logger().info("Loaded trained model for evaluation")

        # Create results directory if it doesn't exist
        results_dir = os.path.join(base_path, 'start_reinforcement_learning', 'evaluation_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize metrics tracker
        metrics = EvaluationMetrics(
            save_dir=results_dir,
            algorithm='mappo',
            map_number=map_number,
            robot_number=robot_number,
            model_episode=model_episode
        )

        PRINT_INTERVAL = 1
        N_GAMES = 100  # Number of evaluation episodes

        # Evaluation loop
        for i in range(N_GAMES):
            # Reset environment and get initial observation
            obs = env.reset()
            # Convert dict -> list of arrays
            list_obs = list(obs.values())
            
            # Get initial unexplored area for exploration ratio calculation
            initial_unexplored = env.get_total_unexplored_area()
            
            # Start tracking metrics for this episode
            metrics.start_episode(initial_unexplored=initial_unexplored)
            
            score = 0
            done = [False] * n_agents
            terminal = [False] * n_agents
            episode_step = 0
            goal_reached = False
            collision = False
            
            # Track start time
            start_time = time.time()
            
            # Track drone positions for trajectory visualization
            robot_positions_history = []
            
            # Truncated means episode has reached max number of steps, done means collided or reached goal
            while not any(terminal):
                # Record robot positions
                robot_positions_history.append(env.get_robot_positions())
                
                # Convert list of observations to global state
                global_state = obs_list_to_state_vector(list_obs)
                
                # Get the actions that the algorithm thinks are best in given observation (decentralized execution)
                # Use deterministic=True for evaluation
                actions, _, _ = mappo_agents.choose_action(obs, global_state, deterministic=True)
                
                # Use step function to get next state and reward info as well as if the episode is 'done'
                obs_, reward, done, truncated, info = env.step(actions)
                
                # Update coordination metrics
                metrics.update_coordination_metrics(
                    env.get_robot_positions(), 
                    exploration_overlap=info['exploration_overlap'] if 'exploration_overlap' in info else 0
                )
                
                # Convert dict -> list of arrays
                list_done = list(done.values())
                list_reward = list(reward.values())
                list_obs_ = list(obs_.values())
                list_trunc = list(truncated.values())
                
                # Check if episode is done
                terminal = [d or t for d, t in zip(list_done, list_trunc)]
                
                # Check for goal or collision
                if any(list_done):
                    # Check if any robot reached the goal
                    for robot_idx in range(n_agents):
                        if env.hasReachedGoal(list_obs[robot_idx], robot_idx):
                            goal_reached = True
                            break
                    
                    # If not goal, must be collision
                    if not goal_reached:
                        collision = True

                # Set new obs to current obs
                obs = obs_
                list_obs = list_obs_
                score += sum(list_reward)
                episode_step += 1
            
            # Record end time and calculate duration
            episode_duration = time.time() - start_time
            
            # Get final unexplored area
            final_unexplored = env.get_total_unexplored_area()
            
            # End episode tracking
            metrics.end_episode(
                reward=score/robot_number,
                steps=episode_step,
                final_unexplored=final_unexplored,
                goal_reached=goal_reached,
                collision=collision
            )
                    
            if i % PRINT_INTERVAL == 0:
                self.get_logger().info('Evaluation Episode: {}, Score: {:.1f}, Steps: {}, Outcome: {}'.format(
                    i, score / robot_number, episode_step, 
                    "Goal Reached" if goal_reached else ("Collision" if collision else "Timeout")
                ))

        # Final evaluation results
        summary = metrics.get_summary_metrics()
        self.get_logger().info(f"Evaluation complete. Final average score: {summary['avg_reward']:.2f}")
        self.get_logger().info(f"Success rate: {summary['success_rate']:.1f}%, " +
                              f"Collision rate: {summary['collision_rate']:.1f}%, " +
                              f"Timeout rate: {summary['timeout_rate']:.1f}%")
        
        # Save metrics to CSV
        metrics.save_metrics_to_csv()
        
        # Generate plots and HTML report
        metrics.plot_metrics()
        report_path = metrics.generate_html_report()
        self.get_logger().info(f"Generated evaluation report: {report_path}")


def main(args=None):
    rclpy.init(args=args)
    
    map_number = int(os.getenv('map_number', '1'))
    robot_number = int(os.getenv('robot_number', '3'))
    model_episode = int(os.getenv('model_episode', '0'))  # 0 means use best model, otherwise use periodic save
    
    node = MAPPOEvaluateNode(map_number, robot_number)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
