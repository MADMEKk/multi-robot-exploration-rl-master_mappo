import numpy as np
import torch as T

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, 
            n_actions, n_agents, batch_size, gamma=0.99, gae_lambda=0.95):
        self.mem_size = max_size
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Hybrid approach: Store episodes for GAE but also maintain a fixed-size buffer for stability
        self.episodes = []
        self.max_episodes = 50  # Limit the number of stored episodes to prevent memory issues
        
        # Current episode data
        self.current_episode = {
            'states': [],            # Global states
            'next_states': [],       # Next global states
            'actions': [],           # Actions for each agent
            'rewards': [],           # Rewards for each agent
            'values': [],            # Value estimates for each agent
            'log_probs': [],         # Log probs for each agent
            'dones': [],             # Done flags for each agent
            'individual_obs': [[] for _ in range(n_agents)],       # Individual observations
            'individual_next_obs': [[] for _ in range(n_agents)]   # Next individual observations
        }
        
        # Track episode length for debugging
        self.episode_step = 0
        self.max_episode_steps = 500  # Maximum steps per episode
        
        # Track if we're ready to learn
        self.ready_to_learn = False

    # Stores transition in the current episode
    def store_transition(self, raw_obs, state, action, reward, 
                         raw_obs_, state_, done, values, log_probs):
        
        # Append data to current episode
        self.current_episode['states'].append(state)
        self.current_episode['next_states'].append(state_)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['values'].append(values)
        self.current_episode['log_probs'].append(log_probs)
        self.current_episode['dones'].append(done)
        
        # Store individual observations for each agent
        for agent_idx in range(self.n_agents):
            self.current_episode['individual_obs'][agent_idx].append(raw_obs[agent_idx])
            self.current_episode['individual_next_obs'][agent_idx].append(raw_obs_[agent_idx])
        
        self.episode_step += 1
    
    def finish_episode(self, last_values=None):
        """Call this when episode ends to calculate advantages and store episode"""
        if not self.current_episode['states']:
            return
        
        try:    
            # Convert lists to numpy arrays
            episode_data = {}
            for key in ['states', 'next_states', 'rewards', 'values', 'log_probs', 'dones']:
                episode_data[key] = np.array(self.current_episode[key])
            
            # Handle actions separately as they're nested lists
            actions_array = np.zeros((len(self.current_episode['actions']), self.n_agents))
            for t, action_dict in enumerate(self.current_episode['actions']):
                for i, agent_id in enumerate(action_dict):
                    # Store the action index, not the discretized action
                    # Convert from discretized action back to index
                    linear_vel, angular_vel = action_dict[agent_id]
                    action_idx = linear_vel * 3 + angular_vel
                    actions_array[t, i] = action_idx
            episode_data['actions'] = actions_array
            
            # Convert individual observations
            episode_data['individual_obs'] = []
            episode_data['individual_next_obs'] = []
            for agent_idx in range(self.n_agents):
                episode_data['individual_obs'].append(
                    np.array(self.current_episode['individual_obs'][agent_idx]))
                episode_data['individual_next_obs'].append(
                    np.array(self.current_episode['individual_next_obs'][agent_idx]))
            
            # Calculate advantages using GAE if last_values is provided
            if last_values is not None:
                try:
                    advantages, returns = self.calculate_gae(
                        episode_data['rewards'], 
                        episode_data['values'], 
                        episode_data['dones'], 
                        last_values)
                    episode_data['advantages'] = advantages
                    episode_data['returns'] = returns
                    self.ready_to_learn = True
                except Exception as e:
                    print(f"Error calculating GAE: {e}")
                    # If GAE calculation fails, we'll still store the episode but without advantages
                    # This allows us to continue training even if one episode has issues
                    pass
            
            # Store the processed episode
            self.episodes.append(episode_data)
            
            # Limit the number of stored episodes to prevent memory issues
            if len(self.episodes) > self.max_episodes:
                self.episodes = self.episodes[-self.max_episodes:]
        except Exception as e:
            print(f"Error in finish_episode: {e}")
            # If there's an error, we'll just reset the episode and continue
            pass
        finally:
            # Always reset current episode to prevent data corruption
            self.current_episode = {
                'states': [],
                'next_states': [],
                'actions': [],
                'rewards': [],
                'values': [],
                'log_probs': [],
                'dones': [],
                'individual_obs': [[] for _ in range(self.n_agents)],
                'individual_next_obs': [[] for _ in range(self.n_agents)]
            }
            self.episode_step = 0
        
        # Keep only recent episodes to manage memory
        if len(self.episodes) > 20:  # Keep only the most recent 20 episodes
            self.episodes = self.episodes[-20:]
    
    def calculate_gae(self, rewards, values, dones, last_values):
        """Properly calculate GAE advantages with robust error handling"""
        try:
            # Convert inputs to numpy arrays if they're not already
            if isinstance(rewards, list):
                rewards = np.array(rewards)
            if isinstance(values, list):
                values = np.array(values)
            if isinstance(dones, list):
                dones = np.array(dones)
            if isinstance(last_values, list):
                last_values = np.array(last_values)
                
            # Ensure all inputs are numpy arrays
            rewards = np.asarray(rewards)
            values = np.asarray(values)
            dones = np.asarray(dones)
            last_values = np.asarray(last_values)
            
            # First, determine if we have dict or list/array inputs
            if isinstance(rewards, dict) or (len(rewards) > 0 and isinstance(rewards[0], dict)):
                # Convert dict to numpy array with shape [time_step, agent_idx]
                T_steps = len(rewards)
                rewards_array = np.zeros((T_steps, self.n_agents))
                for t in range(T_steps):
                    for agent_idx, agent_id in enumerate(rewards[t]):
                        rewards_array[t, agent_idx] = rewards[t][agent_id]
                rewards = rewards_array
            elif len(rewards.shape) == 1:
                if len(rewards) % self.n_agents == 0:
                    rewards = rewards.reshape(-1, self.n_agents)
                else:
                    # Otherwise, we need to handle this differently
                    # This might be a flat array of all rewards for all agents over time
                    # Create a properly shaped array
                    proper_time_steps = max(1, len(rewards) // self.n_agents)
                    rewards_array = np.zeros((proper_time_steps, self.n_agents))
                    for t in range(proper_time_steps):
                        for agent_idx in range(self.n_agents):
                            if t * self.n_agents + agent_idx < len(rewards):
                                rewards_array[t, agent_idx] = rewards[t * self.n_agents + agent_idx]
                    rewards = rewards_array
            
            # Similar handling for values
            if isinstance(values, dict) or (len(values) > 0 and isinstance(values[0], dict)):
                T_values = len(values)
                values_array = np.zeros((T_values, self.n_agents))
                for t in range(T_values):
                    for agent_idx, agent_id in enumerate(values[t]):
                        values_array[t, agent_idx] = values[t][agent_id]
                values = values_array
            elif len(values.shape) == 1:
                if len(values) % self.n_agents == 0:
                    values = values.reshape(-1, self.n_agents)
                else:
                    # Create a properly shaped array
                    proper_time_steps = max(1, len(values) // self.n_agents)
                    values_array = np.zeros((proper_time_steps, self.n_agents))
                    for t in range(proper_time_steps):
                        for agent_idx in range(self.n_agents):
                            if t * self.n_agents + agent_idx < len(values):
                                values_array[t, agent_idx] = values[t * self.n_agents + agent_idx]
                    values = values_array
            
            # Similar handling for dones
            if isinstance(dones, dict) or (len(dones) > 0 and isinstance(dones[0], dict)):
                T_dones = len(dones)
                dones_array = np.zeros((T_dones, self.n_agents))
                for t in range(T_dones):
                    for agent_idx, agent_id in enumerate(dones[t]):
                        dones_array[t, agent_idx] = dones[t][agent_id]
                dones = dones_array
            elif len(dones.shape) == 1:
                if len(dones) % self.n_agents == 0:
                    dones = dones.reshape(-1, self.n_agents)
                else:
                    # Create a properly shaped array
                    proper_time_steps = max(1, len(dones) // self.n_agents)
                    dones_array = np.zeros((proper_time_steps, self.n_agents))
                    for t in range(proper_time_steps):
                        for agent_idx in range(self.n_agents):
                            if t * self.n_agents + agent_idx < len(dones):
                                dones_array[t, agent_idx] = dones[t * self.n_agents + agent_idx]
                    dones = dones_array
            
            # Get the number of time steps from the rewards array shape
            T_steps = rewards.shape[0]
            
            # Initialize advantages and returns with the correct shape
            advantages = np.zeros_like(rewards)
            returns = np.zeros_like(rewards)
            
            # Reshape last_values to ensure it's 2D with shape [1, n_agents]
            if len(last_values.shape) == 0:  # It's a scalar
                last_values = np.full((1, self.n_agents), last_values)
            elif len(last_values.shape) == 1:  # It's a 1D array
                if len(last_values) == 1:  # Single value
                    last_values = np.full((1, self.n_agents), last_values[0])
                else:  # Multiple values
                    # Ensure we have n_agents values
                    if len(last_values) < self.n_agents:
                        # Pad with zeros
                        padded = np.zeros(self.n_agents)
                        padded[:len(last_values)] = last_values
                        last_values = padded
                    elif len(last_values) > self.n_agents:
                        # Truncate
                        last_values = last_values[:self.n_agents]
                    # Reshape to [1, n_agents]
                    last_values = last_values.reshape(1, -1)
            
            # Make sure values has at least T_steps entries
            if len(values.shape) == 1:
                # If values is 1D, reshape it to 2D [T_steps, n_agents]
                values = np.tile(values.reshape(1, -1), (T_steps, 1))
            elif values.shape[0] < T_steps:
                # Handle the case where values array is shorter than rewards
                values_extended = np.zeros((T_steps, self.n_agents))
                values_extended[:values.shape[0]] = values
                values = values_extended
            
            # Make sure dones has the right shape
            if len(dones.shape) == 1 and dones.shape[0] < T_steps * self.n_agents:
                # Pad dones if needed
                padded_dones = np.zeros((T_steps, self.n_agents))
                for t in range(min(T_steps, len(dones))):
                    padded_dones[t, 0] = dones[t]
                dones = padded_dones
            
            # Now calculate GAE
            for agent_idx in range(self.n_agents):
                gae = 0
                for t in reversed(range(T_steps)):
                    try:
                        # Get next value
                        if t == T_steps - 1:
                            next_value = last_values[0, agent_idx]
                        else:
                            next_value = values[t + 1, agent_idx]
                        
                        # Get next non-terminal flag
                        if len(dones.shape) == 1:
                            next_non_terminal = 1.0 - dones[t] if t < len(dones) else 1.0
                        else:
                            next_non_terminal = 1.0 - dones[t, agent_idx]
                        
                        # Calculate delta and GAE
                        delta = rewards[t, agent_idx] + self.gamma * next_value * next_non_terminal - values[t, agent_idx]
                        gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                        advantages[t, agent_idx] = gae
                        returns[t, agent_idx] = gae + values[t, agent_idx]
                    except Exception as e:
                        print(f"Error in GAE calculation at t={t}, agent={agent_idx}: {e}")
                        # Use default values
                        advantages[t, agent_idx] = 0.0
                        returns[t, agent_idx] = values[t, agent_idx]
            
            return advantages, returns
        except Exception as e:
            print(f"Error in calculate_gae: {e}")
            # Create dummy arrays with the right shape
            if isinstance(rewards, np.ndarray) and len(rewards.shape) == 2:
                dummy_shape = rewards.shape
            else:
                dummy_shape = (1, self.n_agents)
            return np.zeros(dummy_shape), np.zeros(dummy_shape)

    def sample_buffer(self):
        """Sample from stored episodes"""
        if not self.episodes or len(self.episodes) < 1:
            return None
            
        # Flatten all episodes into one big batch
        all_states = []
        all_actions = []
        all_advantages = []
        all_returns = []
        all_log_probs = []
        all_individual_obs = [[] for _ in range(self.n_agents)]
        
        for episode in self.episodes:
            if 'advantages' not in episode:
                continue
                
            all_states.extend(episode['states'])
            all_actions.extend(episode['actions'])
            all_advantages.extend(episode['advantages'])
            all_returns.extend(episode['returns'])
            all_log_probs.extend(episode['log_probs'])
            
            for agent_idx in range(self.n_agents):
                all_individual_obs[agent_idx].extend(episode['individual_obs'][agent_idx])
        
        # Convert to numpy arrays
        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        all_advantages = np.array(all_advantages)
        all_returns = np.array(all_returns)
        all_log_probs = np.array(all_log_probs)
        all_individual_obs = [np.array(obs) for obs in all_individual_obs]
        
        total_samples = len(all_states)
        if total_samples < self.batch_size:
            # If we don't have enough samples, return None
            return None
            
        # Sample random indices
        indices = np.random.choice(total_samples, min(self.batch_size, total_samples), replace=False)
        
        # Create batch data
        batch_data = {
            'states': all_states[indices],
            'actions': all_actions[indices],
            'advantages': all_advantages[indices],
            'returns': all_returns[indices], 
            'log_probs': all_log_probs[indices],
            'individual_obs': [obs[indices] for obs in all_individual_obs]
        }
        
        return batch_data

    # Checks if enough episodes have been collected for learning
    def ready(self):
        # Check if we have enough processed episodes with advantages
        if self.ready_to_learn and len(self.episodes) > 0 and any('advantages' in episode for episode in self.episodes):
            return True
        return False
