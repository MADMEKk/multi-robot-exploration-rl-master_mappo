#!/usr/bin/env python3

import os
import json
import argparse
from tabulate import tabulate
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Find and evaluate the best trained models')
    parser.add_argument('--map', type=int, default=1, help='Map number (default: 1)')
    parser.add_argument('--robots', type=int, default=2, help='Number of robots (default: 2)')
    parser.add_argument('--algorithm', type=str, default='mappo', choices=['mappo', 'maddpg'], help='RL algorithm to use (default: mappo)')
    parser.add_argument('--list', action='store_true', help='List all available models')
    parser.add_argument('--evaluate', type=str, help='Evaluate a specific model by its key (e.g., best_score_123.45_ep500)')
    parser.add_argument('--best', action='store_true', help='Evaluate the best model')
    
    args = parser.parse_args()
    
    # Find the model tracker file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    weights_dir = os.path.join(base_dir, 'start_reinforcement_learning', 'deep_learning_weights', args.algorithm)
    
    tracker_file = os.path.join(weights_dir, f'model_tracker_map{args.map}_robots{args.robots}.json')
    
    if not os.path.exists(tracker_file):
        print(f"No tracker file found for map {args.map} with {args.robots} robots.")
        print(f"Expected location: {tracker_file}")
        sys.exit(1)
    
    # Load the tracker data
    with open(tracker_file, 'r') as f:
        tracker_data = json.load(f)
    
    if args.list:
        models = tracker_data['models']
        if not models:
            print("No models found in the tracker file.")
            return
        
        # Sort models by goal success rate and then by score
        sorted_models = sorted(
            models.items(), 
            key=lambda x: (x[1].get('goal_success_rate', 0), x[1].get('score', 0)), 
            reverse=True
        )
        
        print(f"\nAvailable {args.algorithm.upper()} models for map {args.map} with {args.robots} robots:\n")
        
        table_data = []
        for key, info in sorted_models:
            # Mark the best model with an asterisk
            is_best = (key == tracker_data.get('best_model', ''))
            model_name = f"{key} *" if is_best else key
            
            table_data.append([
                model_name,
                info.get('episode', 'N/A'),
                f"{info.get('score', 'N/A'):.2f}" if isinstance(info.get('score'), (int, float)) else info.get('score', 'N/A'),
                f"{info.get('goal_success_rate', 'N/A'):.2f}" if isinstance(info.get('goal_success_rate'), (int, float)) else info.get('goal_success_rate', 'N/A'),
                info.get('timestamp', 'N/A'),
                info.get('path', 'N/A')
            ])
        
        print(tabulate(
            table_data, 
            headers=["Model", "Episode", "Score", "Goal Success Rate", "Timestamp", "Path"],
            tablefmt="grid"
        ))
        
        print("\n* Marks the current best model based on average score")
        print("\nTo evaluate a model, use:")
        print(f"  python find_best_model.py --map {args.map} --robots {args.robots} --algorithm {args.algorithm} --evaluate <model_key>")
        print("Or to evaluate the best model:")
        print(f"  python find_best_model.py --map {args.map} --robots {args.robots} --algorithm {args.algorithm} --best")
        
    elif args.evaluate:
        model_key = args.evaluate
        if model_key not in tracker_data['models']:
            print(f"Model '{model_key}' not found in the tracker file.")
            print("Use --list to see available models.")
            return
        
        model_path = tracker_data['models'][model_key]['path']
        if not os.path.exists(model_path):
            print(f"Model directory not found: {model_path}")
            return
        
        print(f"Evaluating model: {model_key}")
        print(f"Model path: {model_path}")
        print(f"Score: {tracker_data['models'][model_key].get('score', 'N/A')}")
        print(f"Goal success rate: {tracker_data['models'][model_key].get('goal_success_rate', 'N/A')}")
        
        # Extract directory name to use as model_episode
        dir_name = os.path.basename(model_path)
        if dir_name.startswith('periodic_ep'):
            model_episode = dir_name.replace('periodic_ep', '')
        elif dir_name.startswith('best_score_'):
            # Extract the episode number from best_score_XXX.XX_epYYY
            parts = dir_name.split('_ep')
            if len(parts) > 1:
                model_episode = parts[1]
            else:
                model_episode = "0"  # Default if can't parse
        else:
            model_episode = "0"  # Default
            
        print(f"\nTo run this model with ROS2, use these commands:")
        print("\n# First terminal (environment):")
        print(f"ros2 launch start_rl_environment main.launch.py map_number:={args.map} robot_number:={args.robots} headless:=false fast_training:=false")
        print("\n# Second terminal (model evaluation):")
        print(f"export map_number={args.map}")
        print(f"export robot_number={args.robots}")
        print(f"export model_episode={model_episode}")
        print(f"export model_path=\"{model_path}\"")
        print(f"ros2 launch start_reinforcement_learning evaluate_{args.algorithm}.launch.py")
        
    elif args.best:
        best_model_key = tracker_data.get('best_model')
        if not best_model_key:
            print("No best model found in the tracker file.")
            return
            
        print(f"Best {args.algorithm.upper()} model: {best_model_key}")
        print(f"Score: {tracker_data['models'][best_model_key].get('score', 'N/A')}")
        print(f"Goal success rate: {tracker_data['models'][best_model_key].get('goal_success_rate', 'N/A')}")
        
        model_path = tracker_data['models'][best_model_key]['path']
        if not os.path.exists(model_path):
            print(f"Model directory not found: {model_path}")
            return
            
        # Extract directory name to use as model_episode
        dir_name = os.path.basename(model_path)
        if dir_name.startswith('periodic_ep'):
            model_episode = dir_name.replace('periodic_ep', '')
        elif dir_name.startswith('best_score_'):
            # Extract the episode number from best_score_XXX.XX_epYYY
            parts = dir_name.split('_ep')
            if len(parts) > 1:
                model_episode = parts[1]
            else:
                model_episode = "0"  # Default if can't parse
        else:
            model_episode = "0"  # Default
            
        print(f"\nTo run this model with ROS2, use these commands:")
        print("\n# First terminal (environment):")
        print(f"ros2 launch start_rl_environment main.launch.py map_number:={args.map} robot_number:={args.robots} headless:=false fast_training:=false")
        print("\n# Second terminal (model evaluation):")
        print(f"export map_number={args.map}")
        print(f"export robot_number={args.robots}")
        print(f"export model_episode={model_episode}")
        print(f"export model_path=\"{model_path}\"")
        print(f"ros2 launch start_reinforcement_learning evaluate_{args.algorithm}.launch.py")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
