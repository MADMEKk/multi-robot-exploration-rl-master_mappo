import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import csv
import math

class EvaluationMetrics:
    """
    A class for tracking and analyzing metrics during evaluation of the trained models.
    This class provides methods to track success/failure rates, exploration efficiency,
    coordination metrics, and other performance indicators.
    """
    def __init__(self, save_dir='evaluation_results', algorithm='mappo', map_number=1, robot_number=3, model_episode=0):
        """
        Initialize the evaluation metrics tracker.
        
        Args:
            save_dir: Directory to save results
            algorithm: Algorithm name (e.g., 'mappo', 'maddpg')
            map_number: Map used for evaluation
            robot_number: Number of robots
            model_episode: Episode number of the model being evaluated (0 for best model)
        """
        self.save_dir = save_dir
        self.algorithm = algorithm
        self.map_number = map_number
        self.robot_number = robot_number
        self.model_episode = model_episode
        
        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize metrics trackers
        self.reset_metrics()
        
    def reset_metrics(self):
        """Reset all metrics for a new evaluation run."""
        # Episode outcomes
        self.episode_rewards = []
        self.episode_steps = []
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.total_episodes = 0
        
        # Exploration metrics
        self.exploration_ratios = []
        self.initial_unexplored_areas = []
        self.final_unexplored_areas = []
        
        # Coordination metrics
        self.robot_distances = []
        self.exploration_overlaps = []
        self.avg_robot_distances = []
        
        # Time metrics
        self.episode_times = []
        self.start_time = None
        
    def start_episode(self, initial_unexplored=None):
        """
        Call at the start of each evaluation episode.
        
        Args:
            initial_unexplored: Initial unexplored area in the environment
        """
        self.start_time = datetime.now()
        self.total_episodes += 1
        
        # Store the initial unexplored area if provided
        if initial_unexplored is not None:
            self.initial_unexplored_areas.append(initial_unexplored)
    
    def end_episode(self, reward, steps, final_unexplored=None, goal_reached=False, collision=False):
        """
        Call at the end of each evaluation episode.
        
        Args:
            reward: Total reward obtained in the episode
            steps: Number of steps taken in the episode
            final_unexplored: Final unexplored area in the environment
            goal_reached: Whether the goal was reached
            collision: Whether a collision occurred
        """
        # Calculate episode duration
        if self.start_time:
            episode_duration = (datetime.now() - self.start_time).total_seconds()
            self.episode_times.append(episode_duration)
        
        # Store episode outcome
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        
        # Determine episode result
        if goal_reached:
            self.success_count += 1
        elif collision:
            self.collision_count += 1
        else:
            self.timeout_count += 1
            
        # Calculate and store exploration ratio if data is available
        if final_unexplored is not None and len(self.initial_unexplored_areas) > 0:
            initial = self.initial_unexplored_areas[-1]
            exploration_ratio = 1.0 - (final_unexplored / initial) if initial > 0 else 0
            self.exploration_ratios.append(exploration_ratio)
            self.final_unexplored_areas.append(final_unexplored)
    
    def update_coordination_metrics(self, robot_positions, exploration_overlap=0):
        """
        Update metrics related to robot coordination.
        
        Args:
            robot_positions: List of (x, y) positions of each robot
            exploration_overlap: Measure of overlapping exploration effort
        """
        # Calculate distances between all pairs of robots
        distances = []
        for i in range(len(robot_positions)):
            for j in range(i+1, len(robot_positions)):
                x1, y1 = robot_positions[i]
                x2, y2 = robot_positions[j]
                dist = math.hypot(x2 - x1, y2 - y1)
                distances.append(dist)
        
        # Store average distance if there are multiple robots
        if distances:
            avg_distance = sum(distances) / len(distances)
            self.robot_distances.extend(distances)
            self.avg_robot_distances.append(avg_distance)
            
        # Store exploration overlap
        self.exploration_overlaps.append(exploration_overlap)
    
    def get_summary_metrics(self):
        """
        Calculate and return summary metrics.
        
        Returns:
            Dictionary containing summary metrics
        """
        total_episodes = max(self.total_episodes, 1)  # Avoid division by zero
        
        # Calculate rates
        success_rate = self.success_count / total_episodes * 100
        collision_rate = self.collision_count / total_episodes * 100
        timeout_rate = self.timeout_count / total_episodes * 100
        
        # Calculate averages
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        avg_steps = np.mean(self.episode_steps) if self.episode_steps else 0
        avg_time = np.mean(self.episode_times) if self.episode_times else 0
        avg_exploration = np.mean(self.exploration_ratios) if self.exploration_ratios else 0
        avg_robot_distance = np.mean(self.avg_robot_distances) if self.avg_robot_distances else 0
        avg_overlap = np.mean(self.exploration_overlaps) if self.exploration_overlaps else 0
        
        return {
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'avg_time': avg_time,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'timeout_rate': timeout_rate,
            'avg_exploration_ratio': avg_exploration,
            'avg_robot_distance': avg_robot_distance,
            'avg_exploration_overlap': avg_overlap,
            'total_episodes': total_episodes
        }
    
    def save_metrics_to_csv(self):
        """Save detailed metrics to CSV files."""
        # Generate base filename
        base_filename = f"{self.algorithm}_map{self.map_number}_robots{self.robot_number}"
        if self.model_episode > 0:
            base_filename += f"_ep{self.model_episode}"
        else:
            base_filename += "_best"
        
        # Save episode data
        episode_data_file = os.path.join(self.save_dir, f"{base_filename}_episode_data_{self.timestamp}.csv")
        
        with open(episode_data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            header = ['Episode', 'Reward', 'Steps', 'Time', 'Outcome']
            if self.exploration_ratios:
                header.extend(['Exploration_Ratio', 'Initial_Unexplored', 'Final_Unexplored'])
            if self.avg_robot_distances:
                header.extend(['Avg_Robot_Distance', 'Exploration_Overlap'])
            writer.writerow(header)
            
            # Write episode data
            for i in range(self.total_episodes):
                row = [i+1]
                # Add basic metrics
                if i < len(self.episode_rewards):
                    row.append(self.episode_rewards[i])
                else:
                    row.append('')
                    
                if i < len(self.episode_steps):
                    row.append(self.episode_steps[i])
                else:
                    row.append('')
                    
                if i < len(self.episode_times):
                    row.append(self.episode_times[i])
                else:
                    row.append('')
                
                # Determine outcome (success, collision, timeout)
                if i < self.success_count:
                    outcome = 'Success'
                elif i < self.success_count + self.collision_count:
                    outcome = 'Collision'
                else:
                    outcome = 'Timeout'
                row.append(outcome)
                
                # Add exploration metrics if available
                if self.exploration_ratios and i < len(self.exploration_ratios):
                    row.append(self.exploration_ratios[i])
                    if i < len(self.initial_unexplored_areas):
                        row.append(self.initial_unexplored_areas[i])
                    else:
                        row.append('')
                    if i < len(self.final_unexplored_areas):
                        row.append(self.final_unexplored_areas[i])
                    else:
                        row.append('')
                
                # Add coordination metrics if available
                if self.avg_robot_distances and i < len(self.avg_robot_distances):
                    row.append(self.avg_robot_distances[i])
                    if i < len(self.exploration_overlaps):
                        row.append(self.exploration_overlaps[i])
                    else:
                        row.append('')
                
                writer.writerow(row)
        
        # Save summary metrics
        summary = self.get_summary_metrics()
        summary_file = os.path.join(self.save_dir, f"{base_filename}_summary_{self.timestamp}.csv")
        
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in summary.items():
                writer.writerow([key, value])
        
        return episode_data_file, summary_file
    
    def plot_metrics(self):
        """Generate and save plots for visualizing the metrics."""
        # Generate base filename
        base_filename = f"{self.algorithm}_map{self.map_number}_robots{self.robot_number}"
        if self.model_episode > 0:
            base_filename += f"_ep{self.model_episode}"
        else:
            base_filename += "_best"
        
        # Plot reward over episodes
        if self.episode_rewards:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.episode_rewards) + 1), self.episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title(f'{self.algorithm.upper()} Rewards per Episode')
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, f"{base_filename}_rewards_{self.timestamp}.png"))
            plt.close()
        
        # Plot episode steps
        if self.episode_steps:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.episode_steps) + 1), self.episode_steps)
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.title(f'{self.algorithm.upper()} Steps per Episode')
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, f"{base_filename}_steps_{self.timestamp}.png"))
            plt.close()
        
        # Plot exploration ratios
        if self.exploration_ratios:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.exploration_ratios) + 1), self.exploration_ratios)
            plt.xlabel('Episode')
            plt.ylabel('Exploration Ratio')
            plt.title(f'{self.algorithm.upper()} Exploration Efficiency')
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, f"{base_filename}_exploration_{self.timestamp}.png"))
            plt.close()
        
        # Plot outcome distribution as pie chart
        labels = ['Success', 'Collision', 'Timeout']
        sizes = [self.success_count, self.collision_count, self.timeout_count]
        if sum(sizes) > 0:  # Only create pie chart if we have data
            plt.figure(figsize=(8, 8))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title(f'{self.algorithm.upper()} Episode Outcomes')
            plt.savefig(os.path.join(self.save_dir, f"{base_filename}_outcomes_{self.timestamp}.png"))
            plt.close()
        
        # Plot coordination metrics
        if self.avg_robot_distances:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.avg_robot_distances) + 1), self.avg_robot_distances)
            plt.xlabel('Episode')
            plt.ylabel('Average Distance Between Robots')
            plt.title(f'{self.algorithm.upper()} Robot Coordination')
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, f"{base_filename}_coordination_{self.timestamp}.png"))
            plt.close()
    
    def generate_html_report(self):
        """Generate an HTML report with the evaluation results."""
        # Generate base filename
        base_filename = f"{self.algorithm}_map{self.map_number}_robots{self.robot_number}"
        if self.model_episode > 0:
            base_filename += f"_ep{self.model_episode}"
        else:
            base_filename += "_best"
            
        summary = self.get_summary_metrics()
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.algorithm.upper()} Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 80%; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #2c3e50; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .metric-card {{ background-color: #f8f9fa; padding: 15px; margin: 10px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-title {{ color: #7f8c8d; font-size: 14px; }}
                .row {{ display: flex; flex-wrap: wrap; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            </style>
        </head>
        <body>
            <h1>{self.algorithm.upper()} Evaluation Report</h1>
            
            <div class="summary">
                <h2>Evaluation Summary</h2>
                <p>Algorithm: {self.algorithm.upper()}</p>
                <p>Map: {self.map_number}</p>
                <p>Robots: {self.robot_number}</p>
                <p>Model Episode: {'Best Model' if self.model_episode == 0 else self.model_episode}</p>
                <p>Total Episodes Evaluated: {self.total_episodes}</p>
                <p>Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <h2>Key Performance Metrics</h2>
            <div class="row">
                <div class="metric-card">
                    <div class="metric-value">{summary['avg_reward']:.2f}</div>
                    <div class="metric-title">Average Reward</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary['success_rate']:.1f}%</div>
                    <div class="metric-title">Success Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary['collision_rate']:.1f}%</div>
                    <div class="metric-title">Collision Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary['avg_steps']:.1f}</div>
                    <div class="metric-title">Average Steps</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary['avg_exploration_ratio']*100:.1f}%</div>
                    <div class="metric-title">Exploration Efficiency</div>
                </div>
            </div>
            
            <h2>Detailed Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """
        
        # Add each metric to the table
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                html_content += f"""
                <tr>
                    <td>{key.replace('_', ' ').title()}</td>
                    <td>{formatted_value}</td>
                </tr>
                """
        
        # Close the table and add visualizations
        html_content += """
            </table>
            
            <h2>Visualizations</h2>
        """
        
        # Add images
        image_types = ['rewards', 'steps', 'exploration', 'outcomes', 'coordination']
        for img_type in image_types:
            img_path = f"{base_filename}_{img_type}_{self.timestamp}.png"
            if os.path.exists(os.path.join(self.save_dir, img_path)):
                html_content += f"""
                <h3>{img_type.replace('_', ' ').title()}</h3>
                <img src="{img_path}" alt="{img_type} visualization">
                """
        
        # Close the HTML
        html_content += """
        </body>
        </html>
        """
        
        # Save the HTML report
        report_path = os.path.join(self.save_dir, f"{base_filename}_report_{self.timestamp}.html")
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path 