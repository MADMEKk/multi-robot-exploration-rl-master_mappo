import torch as T
import torch.nn.functional as F
from start_reinforcement_learning.mappo_algorithm.networks import ActorNetwork, CriticNetwork
import numpy as np

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, agent_idx, chkpt_dir,
                 alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.95, 
                 gae_lambda=0.95, clip_param=0.2, entropy_coef=0.01, value_coef=0.5):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        
        # Actor uses individual observation space (decentralized execution)
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir, name=self.agent_name+'_actor')
        # Critic uses global state space (centralized training)
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2,
                                    chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')

    def choose_action(self, individual_obs, global_state=None, deterministic=False):
        # Actor uses individual observation
        individual_state = T.tensor(individual_obs[np.newaxis, :], dtype=T.float,
                                   device=self.actor.device)
        
        action, log_prob = self.actor.sample_action(individual_state, deterministic)
        
        # If global state is provided, use it for value estimation
        value = None
        if global_state is not None:
            global_state_tensor = T.tensor(global_state[np.newaxis, :], dtype=T.float,
                                          device=self.critic.device)
            value = self.critic.forward(global_state_tensor)
            value = value.detach().cpu().numpy()[0]
        
        return action.detach().cpu().numpy()[0], value, log_prob.detach().cpu().numpy()[0]
    
    def evaluate_actions(self, individual_states, global_states, actions):
        # Actor evaluation uses individual observations
        action_log_probs, entropy = self.actor.evaluate_actions(individual_states, actions)
        
        # Critic evaluation uses global state
        values = self.critic.forward(global_states).squeeze(-1)
        
        return values, action_log_probs, entropy
    
    def discretize(self, action_index):
        # Map from action index to discrete linear and angular velocity actions
        # For 9 actions (3 linear x 3 angular)
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
    
    def update(self, individual_states, global_states, actions, old_log_probs, advantages, returns):
        # Convert numpy arrays to tensors
        individual_states = T.tensor(individual_states, dtype=T.float, device=self.actor.device)
        global_states = T.tensor(global_states, dtype=T.float, device=self.critic.device)
        actions = T.tensor(actions, dtype=T.long, device=self.actor.device)
        old_log_probs = T.tensor(old_log_probs, dtype=T.float, device=self.actor.device)
        advantages = T.tensor(advantages, dtype=T.float, device=self.actor.device)
        returns = T.tensor(returns, dtype=T.float, device=self.critic.device)
        
        # Normalize advantages (helps with training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Evaluate actions
        values, new_log_probs, entropy = self.evaluate_actions(individual_states, global_states, actions)
        
        # Calculate ratio and clipped ratio
        ratio = T.exp(new_log_probs - old_log_probs)
        clipped_ratio = T.clamp(ratio, 1-self.clip_param, 1+self.clip_param)
        
        # Calculate actor and critic losses
        policy_loss = -T.min(ratio * advantages, clipped_ratio * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Update actor and critic networks
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy_loss.item()
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
