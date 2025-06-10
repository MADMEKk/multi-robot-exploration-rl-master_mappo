import torch as T
import torch.nn.functional as F
from start_reinforcement_learning.mappo_algorithm.agent import Agent
import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)

class MAPPO:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='robot', alpha=0.01, beta=0.01, fc1=512, 
                 fc2=512, gamma=0.99, gae_lambda=0.95, clip_param=0.2,
                 entropy_coef=0.01, value_coef=0.5, tau=0.01, 
                 chkpt_dir='tmp/mappo/', node_logger=None):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.logger = node_logger
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = 4  # Number of epochs to update policy per batch
        
        # Path now includes map and robot numbers from mappo_main.py
        
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir, fc1=fc1, fc2=fc2, gamma=gamma,
                            gae_lambda=gae_lambda, clip_param=clip_param,
                            entropy_coef=entropy_coef, value_coef=value_coef))

    def save_checkpoint(self):
        if self.logger:
            self.logger.get_logger().info('... saving checkpoint to dynamic path based on map and robot count ...')
        else:
            print('... saving checkpoint to dynamic path based on map and robot count ...')
        
        try:
            for agent in self.agents:
                agent.save_models()
            if self.logger:
                self.logger.get_logger().info('Successfully saved all agent models')
            else:
                print('Successfully saved all agent models')
        except Exception as e:
            if self.logger:
                self.logger.get_logger().error(f'Error saving models: {str(e)}')
            else:
                print(f'Error saving models: {str(e)}')

    def load_checkpoint(self):
        if self.logger:
            self.logger.get_logger().info('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()
    
    # Convert action index to discrete robot actions
    def discretize(self, action_index):
        # Map from action index to discrete linear and angular velocity actions
        linear_idx = action_index // 3
        angular_idx = action_index % 3
        
        if linear_idx == 0:
            linear_velocity_action = 0  # Stop
        elif linear_idx == 1:
            linear_velocity_action = 1  # Medium speed
        else:
            linear_velocity_action = 2  # Full speed
            
        if angular_idx == 0:
            angular_velocity_action = 0  # Turn left
        elif angular_idx == 1:
            angular_velocity_action = 1  # Go straight
        else:
            angular_velocity_action = 2  # Turn right
            
        discrete_actions = np.array([linear_velocity_action, angular_velocity_action])
        return discrete_actions
                
    # Returns dict of each agent's chosen action for linear and angular velocity
    def choose_action(self, raw_obs, global_state=None, deterministic=False):        
        actions = {}
        values = {}
        log_probs = {}
        
        for agent_idx, agent_id in enumerate(raw_obs):
            # For each agent, get its individual observation
            individual_obs = raw_obs[agent_id]
            
            # Choose action using the agent's policy (decentralized execution)
            action_index, value, log_prob = self.agents[agent_idx].choose_action(individual_obs, global_state, deterministic)
            
            # Convert action index to discrete actions for the environment
            discrete_actions = self.discretize(action_index)
            
            # Store the results
            actions[agent_id] = discrete_actions
            values[agent_id] = value
            log_probs[agent_id] = log_prob
            
        return actions, values, log_probs

    # Adjusts actor and critic weights using PPO algorithm
    def learn(self, memory):
        # If memory doesn't have enough data, return
        if not memory.ready():
            return
        
        # Sample batch from memory
        batch_data = memory.sample_buffer()
        if batch_data is None:
            return
            
        # Update each agent
        for agent_idx, agent in enumerate(self.agents):
            # Get data for this agent
            individual_states = batch_data['individual_obs'][agent_idx]
            global_states = batch_data['states']
            actions = batch_data['actions'][:, agent_idx]
            
            # Carefully extract log probabilities
            log_probs = batch_data['log_probs']
            try:
                if log_probs.ndim == 1:
                    old_log_probs = log_probs  # No indexing needed, single agent
                else:
                    old_log_probs = log_probs[:, agent_idx]
                    
                # Ensure log_probs are in a valid format
                if isinstance(old_log_probs, np.ndarray) and old_log_probs.dtype == np.dtype('O'):
                    # Convert object array to float array
                    old_log_probs = np.array([float(x) if x is not None else 0.0 
                                             for x in old_log_probs], dtype=np.float32)
            except Exception as e:
                if self.logger:
                    self.logger.get_logger().warning(f"Error processing log_probs: {e}")
                # Create a default array of zeros as fallback
                old_log_probs = np.zeros(len(individual_states), dtype=np.float32)
                
            advantages = batch_data['advantages'][:, agent_idx]
            returns = batch_data['returns'][:, agent_idx]
            
            # Multiple epochs of updates (typical in PPO)
            for _ in range(self.update_epochs):
                # Update agent networks
                policy_loss, value_loss, entropy_loss = agent.update(
                    individual_states, global_states, actions, old_log_probs, advantages, returns
                )
            
            if self.logger:
                self.logger.get_logger().info(
                    f"Agent {agent_idx} - Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy_loss:.4f}"
                )
                
    def get_lr(self):
        """
        Get the current learning rate of the agents.
        
        Returns:
            float: The learning rate of the first agent's actor optimizer
        """
        if not self.agents:
            return 0.0
            
        # Get learning rate from the first agent's actor optimizer
        # (All agents likely have the same learning rate)
        try:
            return self.agents[0].actor.optimizer.param_groups[0]['lr']
        except (IndexError, AttributeError):
            return 0.0
