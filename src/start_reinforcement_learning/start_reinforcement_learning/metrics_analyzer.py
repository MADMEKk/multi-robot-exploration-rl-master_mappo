import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os
from datetime import datetime

class MetricsAnalyzer:
    """
    Analyzes and compares performance metrics between MAPPO and MADDPG algorithms.
    """
    def __init__(self, save_dir='comparison_results'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def analyze_rewards(self, mappo_rewards, maddpg_rewards):
        """
        Analyze reward statistics and perform statistical tests
        """
        # Calculate basic statistics
        mappo_mean = np.mean(mappo_rewards)
        maddpg_mean = np.mean(maddpg_rewards)
        mappo_std = np.std(mappo_rewards)
        maddpg_std = np.std(maddpg_rewards)
        mappo_median = np.median(mappo_rewards)
        maddpg_median = np.median(maddpg_rewards)
        
        # Perform t-test to check if difference is statistically significant
        t_stat, p_value = stats.ttest_ind(mappo_rewards, maddpg_rewards)
        
        # Create dataframe for results
        stats_df = pd.DataFrame({
            'Metric': ['Mean Reward', 'Std Dev', 'Median', 't-statistic', 'p-value'],
            'MAPPO': [mappo_mean, mappo_std, mappo_median, t_stat, p_value],
            'MADDPG': [maddpg_mean, maddpg_std, maddpg_median, '', '']
        })
        
        # Save statistics
        stats_df.to_csv(f"{self.save_dir}/reward_statistics_{self.timestamp}.csv", index=False)
        
        return stats_df
    
    def analyze_efficiency(self, mappo_steps, maddpg_steps, mappo_times, maddpg_times):
        """
        Analyze efficiency metrics (steps per episode and execution time)
        """
        # Calculate average steps to completion
        mappo_avg_steps = np.mean(mappo_steps)
        maddpg_avg_steps = np.mean(maddpg_steps)
        
        # Calculate average execution time
        mappo_avg_time = np.mean(mappo_times)
        maddpg_avg_time = np.mean(maddpg_times)
        
        # Calculate steps per second (efficiency)
        mappo_steps_per_sec = np.sum(mappo_steps) / np.sum(mappo_times)
        maddpg_steps_per_sec = np.sum(maddpg_steps) / np.sum(maddpg_times)
        
        # Create dataframe for results
        efficiency_df = pd.DataFrame({
            'Metric': ['Avg Steps', 'Avg Time (s)', 'Steps/Second'],
            'MAPPO': [mappo_avg_steps, mappo_avg_time, mappo_steps_per_sec],
            'MADDPG': [maddpg_avg_steps, maddpg_avg_time, maddpg_steps_per_sec]
        })
        
        # Save statistics
        efficiency_df.to_csv(f"{self.save_dir}/efficiency_metrics_{self.timestamp}.csv", index=False)
        
        return efficiency_df
    
    def analyze_success_rates(self, mappo_goals, maddpg_goals, mappo_collisions, maddpg_collisions, n_episodes):
        """
        Analyze success and failure rates
        """
        # Calculate success rates
        mappo_success_rate = mappo_goals / n_episodes * 100
        maddpg_success_rate = maddpg_goals / n_episodes * 100
        
        # Calculate collision rates
        mappo_collision_rate = mappo_collisions / n_episodes * 100
        maddpg_collision_rate = maddpg_collisions / n_episodes * 100
        
        # Calculate timeout rates
        mappo_timeout_rate = (n_episodes - mappo_goals - mappo_collisions) / n_episodes * 100
        maddpg_timeout_rate = (n_episodes - maddpg_goals - maddpg_collisions) / n_episodes * 100
        
        # Create dataframe for results
        rates_df = pd.DataFrame({
            'Metric': ['Success Rate (%)', 'Collision Rate (%)', 'Timeout Rate (%)'],
            'MAPPO': [mappo_success_rate, mappo_collision_rate, mappo_timeout_rate],
            'MADDPG': [maddpg_success_rate, maddpg_collision_rate, maddpg_timeout_rate]
        })
        
        # Save statistics
        rates_df.to_csv(f"{self.save_dir}/success_rates_{self.timestamp}.csv", index=False)
        
        return rates_df
    
    def plot_learning_curves(self, mappo_rewards, maddpg_rewards, window_size=10):
        """
        Plot smoothed learning curves for both algorithms
        """
        # Create smoothed curves using moving average
        def smooth(data, window_size):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        episodes = range(1, len(mappo_rewards) + 1)
        
        if len(mappo_rewards) > window_size:
            smoothed_mappo = smooth(mappo_rewards, window_size)
            smoothed_maddpg = smooth(maddpg_rewards, window_size)
            smoothed_episodes = range(window_size, len(mappo_rewards) + 1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(smoothed_episodes, smoothed_mappo, label='MAPPO')
            plt.plot(smoothed_episodes, smoothed_maddpg, label='MADDPG')
        else:
            plt.figure(figsize=(10, 6))
            plt.plot(episodes, mappo_rewards, label='MAPPO')
            plt.plot(episodes, maddpg_rewards, label='MADDPG')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Smoothed Reward Curves')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(f"{self.save_dir}/learning_curves_{self.timestamp}.png")
        plt.close()
    
    def plot_reward_distributions(self, mappo_rewards, maddpg_rewards):
        """
        Plot reward distributions as histograms
        """
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(mappo_rewards, bins=20, alpha=0.7, label='MAPPO')
        plt.axvline(np.mean(mappo_rewards), color='r', linestyle='dashed', linewidth=1, label='Mean')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('MAPPO Reward Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(maddpg_rewards, bins=20, alpha=0.7, label='MADDPG')
        plt.axvline(np.mean(maddpg_rewards), color='r', linestyle='dashed', linewidth=1, label='Mean')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('MADDPG Reward Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/reward_distributions_{self.timestamp}.png")
        plt.close()
    
    def plot_step_comparison(self, mappo_steps, maddpg_steps):
        """
        Plot comparison of steps taken per episode
        """
        episodes = range(1, len(mappo_steps) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, mappo_steps, label='MAPPO')
        plt.plot(episodes, maddpg_steps, label='MADDPG')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Steps per Episode')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f"{self.save_dir}/steps_comparison_{self.timestamp}.png")
        plt.close()
    
    def generate_comprehensive_report(self, metrics):
        """
        Generate a comprehensive analysis report
        """
        # Extract metrics
        mappo_rewards = metrics['mappo']['episode_rewards']
        maddpg_rewards = metrics['maddpg']['episode_rewards']
        mappo_steps = metrics['mappo']['episode_steps']
        maddpg_steps = metrics['maddpg']['episode_steps']
        mappo_times = metrics['mappo']['execution_times']
        maddpg_times = metrics['maddpg']['execution_times']
        mappo_goals = metrics['mappo']['goals_reached']
        maddpg_goals = metrics['maddpg']['goals_reached']
        mappo_collisions = metrics['mappo']['collisions']
        maddpg_collisions = metrics['maddpg']['collisions']
        n_episodes = len(mappo_rewards)
        
        # Perform analyses
        reward_stats = self.analyze_rewards(mappo_rewards, maddpg_rewards)
        efficiency_stats = self.analyze_efficiency(mappo_steps, maddpg_steps, mappo_times, maddpg_times)
        success_stats = self.analyze_success_rates(mappo_goals, maddpg_goals, mappo_collisions, maddpg_collisions, n_episodes)
        
        # Generate plots
        self.plot_learning_curves(mappo_rewards, maddpg_rewards)
        self.plot_reward_distributions(mappo_rewards, maddpg_rewards)
        self.plot_step_comparison(mappo_steps, maddpg_steps)
        
        # Generate HTML report
        html_report = f"""
        <html>
        <head>
            <title>MAPPO vs MADDPG Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 80%; margin: 20px 0; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #333366; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>MAPPO vs MADDPG Comparison Report</h1>
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>This report compares the performance of MAPPO (Multi-Agent Proximal Policy Optimization) and 
                MADDPG (Multi-Agent Deep Deterministic Policy Gradient) algorithms on a multi-robot exploration task.</p>
                <p>Number of episodes: {n_episodes}</p>
                <p>Date of analysis: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <h2>Reward Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>MAPPO</th>
                    <th>MADDPG</th>
                </tr>
        """
        
        # Add reward statistics to HTML
        for _, row in reward_stats.iterrows():
            html_report += f"""
                <tr>
                    <td>{row['Metric']}</td>
                    <td>{row['MAPPO']}</td>
                    <td>{row['MADDPG']}</td>
                </tr>
            """
        
        html_report += """
            </table>
            
            <h2>Efficiency Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>MAPPO</th>
                    <th>MADDPG</th>
                </tr>
        """
        
        # Add efficiency metrics to HTML
        for _, row in efficiency_stats.iterrows():
            html_report += f"""
                <tr>
                    <td>{row['Metric']}</td>
                    <td>{row['MAPPO']}</td>
                    <td>{row['MADDPG']}</td>
                </tr>
            """
        
        html_report += """
            </table>
            
            <h2>Success Rates</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>MAPPO</th>
                    <th>MADDPG</th>
                </tr>
        """
        
        # Add success rates to HTML
        for _, row in success_stats.iterrows():
            html_report += f"""
                <tr>
                    <td>{row['Metric']}</td>
                    <td>{row['MAPPO']}</td>
                    <td>{row['MADDPG']}</td>
                </tr>
            """
        
        html_report += f"""
            </table>
            
            <h2>Visualization</h2>
            <h3>Learning Curves</h3>
            <img src="learning_curves_{self.timestamp}.png" alt="Learning Curves">
            
            <h3>Reward Distributions</h3>
            <img src="reward_distributions_{self.timestamp}.png" alt="Reward Distributions">
            
            <h3>Steps Comparison</h3>
            <img src="steps_comparison_{self.timestamp}.png" alt="Steps Comparison">
            
            <h2>Algorithm Comparison Analysis</h2>
            <h3>Key Differences Between MAPPO and MADDPG</h3>
            <ul>
                <li><strong>Architecture:</strong> MAPPO uses a centralized critic with decentralized actors (CTDE), while MADDPG uses a centralized training with decentralized execution approach.</li>
                <li><strong>Policy Update:</strong> MAPPO uses PPO's clipped objective function for stable policy updates, while MADDPG uses deterministic policy gradients.</li>
                <li><strong>Exploration:</strong> MAPPO uses stochastic policies with entropy regularization, while MADDPG uses noise added to deterministic actions.</li>
                <li><strong>Sample Efficiency:</strong> MADDPG is typically more sample-efficient but can be less stable, while MAPPO requires more samples but offers more stable learning.</li>
            </ul>
            
            <h3>Conclusion</h3>
            <p>Based on the metrics analyzed, the following conclusions can be drawn:</p>
            <ul>
                <li><strong>Performance:</strong> {'MAPPO outperforms MADDPG' if np.mean(mappo_rewards) > np.mean(maddpg_rewards) else 'MADDPG outperforms MAPPO'} in terms of average reward.</li>
                <li><strong>Efficiency:</strong> {'MAPPO is more efficient' if np.mean(mappo_steps) < np.mean(maddpg_steps) else 'MADDPG is more efficient'} in terms of steps needed to complete episodes.</li>
                <li><strong>Success Rate:</strong> {'MAPPO has a higher success rate' if mappo_goals > maddpg_goals else 'MADDPG has a higher success rate'} in reaching goals.</li>
                <li><strong>Computational Cost:</strong> {'MAPPO has lower computational cost' if np.mean(mappo_times) < np.mean(maddpg_times) else 'MADDPG has lower computational cost'} based on execution time.</li>
            </ul>
            
            <p>These results suggest that {'MAPPO' if (np.mean(mappo_rewards) > np.mean(maddpg_rewards) and mappo_goals > maddpg_goals) else 'MADDPG'} might be better suited for this particular multi-robot exploration task.</p>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(f"{self.save_dir}/comparison_report_{self.timestamp}.html", 'w') as f:
            f.write(html_report)
        
        return f"{self.save_dir}/comparison_report_{self.timestamp}.html"
