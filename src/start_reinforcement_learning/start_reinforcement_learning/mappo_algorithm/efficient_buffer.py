"""
Memory-efficient implementation of the Multi-Agent Replay Buffer.
This implementation focuses on reduced memory usage and faster operations.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque


class EfficientMultiAgentBuffer:
    """
    Memory-efficient implementation of multi-agent experience replay buffer.
    Uses a circular buffer approach with fixed size arrays for better memory efficiency.
    """
    
    def __init__(self, max_size: int, critic_dims: int, actor_dims: List[int], 
                n_actions: int, n_agents: int, batch_size: int = 2048,
                gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Initialize the multi-agent buffer.
        
        Args:
            max_size: Maximum transitions to store in buffer
            critic_dims: Dimensions of critic input (global state)
            actor_dims: Dimensions of actor inputs (observations) for each agent
            n_actions: Number of possible actions
            n_agents: Number of agents
            batch_size: Size of training batches
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.max_size = max_size
        self.n_agents = n_agents
        self.n_actions = n_actions
        
        # Index management
        self.ptr = 0  # Current position in buffer
        self.size = 0  # Current size of buffer
        self.episode_start = 0  # Start position of current episode

        # Preallocate arrays for transitions
        # Global state
        self.states = np.zeros((max_size, critic_dims), dtype=np.float32)
        self.next_states = np.zeros((max_size, critic_dims), dtype=np.float32)
        
        # Agent-specific arrays
        self.actions = np.zeros((max_size, n_agents), dtype=np.int32)
        self.rewards = np.zeros((max_size, n_agents), dtype=np.float32)
        self.dones = np.zeros((max_size, n_agents), dtype=np.bool_)
        self.values = np.zeros((max_size, n_agents), dtype=np.float32)
        self.log_probs = np.zeros((max_size, n_agents), dtype=np.float32)
        
        # Individual observations for each agent (jagged array)
        self.observations = []
        self.next_observations = []
        for i in range(n_agents):
            self.observations.append(np.zeros((max_size, actor_dims[i]), dtype=np.float32))
            self.next_observations.append(np.zeros((max_size, actor_dims[i]), dtype=np.float32))
        
        # Advantage and return storage
        self.advantages = np.zeros((max_size, n_agents), dtype=np.float32)
        self.returns = np.zeros((max_size, n_agents), dtype=np.float32)
        
        # Episode tracking
        self.episode_lengths = []  # Store lengths of completed episodes
        self.current_episode_length = 0
        self.episode_borders = []  # Store (start, end) indices of episodes
        self.episodes_in_buffer = 0
        
        # Tracking if buffer has enough data
        self.ready_to_learn = False
        
    def store_transition(self, obs: Dict[str, np.ndarray], state: np.ndarray, 
                        action: Dict[str, np.ndarray], reward: Dict[str, float],
                        next_obs: Dict[str, np.ndarray], next_state: np.ndarray, 
                        done: Dict[str, bool], values: Dict[str, float], 
                        log_probs: Dict[str, float]):
        """
        Store a transition in the buffer.
        
        Args:
            obs: Dict mapping agent IDs to observations
            state: Global state
            action: Dict mapping agent IDs to actions
            reward: Dict mapping agent IDs to rewards
            next_obs: Dict mapping agent IDs to next observations
            next_state: Next global state
            done: Dict mapping agent IDs to done flags
            values: Dict mapping agent IDs to value estimates
            log_probs: Dict mapping agent IDs to log probabilities
        """
        # Store global states
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        
        # Store agent-specific data
        for i, agent_id in enumerate(obs):
            # Convert action from discretized format to index
            if isinstance(action[agent_id], np.ndarray) and len(action[agent_id]) > 1:
                linear_vel, angular_vel = action[agent_id]
                action_idx = linear_vel * 3 + angular_vel
            else:
                action_idx = action[agent_id]
                
            self.actions[self.ptr, i] = action_idx
            self.rewards[self.ptr, i] = reward[agent_id]
            self.dones[self.ptr, i] = done[agent_id]
            
            # Handle values and log_probs which might be None
            self.values[self.ptr, i] = values.get(agent_id, 0.0) if isinstance(values, dict) else 0.0
            self.log_probs[self.ptr, i] = log_probs.get(agent_id, 0.0) if isinstance(log_probs, dict) else 0.0
            
            # Store individual observations
            self.observations[i][self.ptr] = obs[agent_id]
            self.next_observations[i][self.ptr] = next_obs[agent_id]
        
        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.current_episode_length += 1
        
    def finish_episode(self, last_values: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None):
        """
        Finish the current episode and calculate advantages.
        
        Args:
            last_values: Value estimates for the final state
        """
        if self.current_episode_length == 0:
            return  # No transitions stored for current episode
            
        episode_end = (self.episode_start + self.current_episode_length - 1) % self.max_size
        episode_slice = self._get_episode_indices(self.episode_start, self.current_episode_length)
        
        # Calculate advantages and returns
        self._calculate_advantages(episode_slice, last_values)
        
        # Record episode data
        self.episode_lengths.append(self.current_episode_length)
        self.episode_borders.append((self.episode_start, episode_end))
        self.episodes_in_buffer += 1
        
        # If we're wrapping around the buffer, remove oldest episodes that would be overwritten
        while self.episodes_in_buffer > 1:
            oldest_start, oldest_end = self.episode_borders[0]
            
            # Check if next episode will overwrite this one
            next_episode_end = (self.ptr + self.current_episode_length - 1) % self.max_size
            
            # If next episode wouldn't overlap with oldest, we're done
            if not self._will_overlap(self.ptr, next_episode_end, oldest_start, oldest_end):
                break
                
            # Remove oldest episode
            self.episode_borders.pop(0)
            self.episode_lengths.pop(0)
            self.episodes_in_buffer -= 1
            
        # Start new episode at current pointer
        self.episode_start = self.ptr
        self.current_episode_length = 0
        
        # Mark buffer as ready to learn once we have enough data
        if sum(self.episode_lengths) >= self.batch_size:
            self.ready_to_learn = True
            
    def _get_episode_indices(self, start: int, length: int) -> np.ndarray:
        """
        Get indices for an episode considering buffer wraparound.
        
        Args:
            start: Starting index of the episode
            length: Length of the episode
            
        Returns:
            Array of indices representing the episode
        """
        indices = np.arange(start, start + length) % self.max_size
        return indices
        
    def _calculate_advantages(self, indices: np.ndarray, 
                            last_values: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None):
        """
        Calculate advantages and returns for the episode using GAE.
        
        Args:
            indices: Indices of the episode
            last_values: Value estimates for final state
        """
        # Convert last_values to appropriate format
        if last_values is None:
            last_values = np.zeros(self.n_agents, dtype=np.float32)
        elif isinstance(last_values, dict):
            last_values_array = np.zeros(self.n_agents, dtype=np.float32)
            for i, key in enumerate(last_values):
                last_values_array[i] = last_values.get(key, 0.0)
            last_values = last_values_array
        elif isinstance(last_values, list):
            last_values = np.array(last_values, dtype=np.float32)
            
        # Make sure last_values has correct shape
        if len(last_values) < self.n_agents:
            # Pad with zeros if needed
            last_values = np.pad(last_values, (0, self.n_agents - len(last_values)), 
                                constant_values=0.0)
        
        # Calculate advantages for each agent
        for agent_idx in range(self.n_agents):
            # Extract relevant data for this agent
            rewards = self.rewards[indices, agent_idx]
            values = self.values[indices, agent_idx]
            dones = self.dones[indices, agent_idx]
            
            # Create arrays for advantages and returns
            advantages = np.zeros_like(rewards)
            returns = np.zeros_like(rewards)
            
            # Initialize with last value if provided
            next_value = last_values[agent_idx]
            next_advantage = 0
            
            # Calculate returns and advantages (backwards)
            for t in reversed(range(len(rewards))):
                # For terminal states, use 0 as next value
                if dones[t]:
                    next_value = 0
                    next_advantage = 0
                    
                # Calculate TD error and return
                td_error = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                advantages[t] = td_error + self.gamma * self.gae_lambda * next_advantage * (1 - dones[t])
                returns[t] = rewards[t] + self.gamma * next_value * (1 - dones[t])
                
                # Update next values
                next_value = values[t]
                next_advantage = advantages[t]
                
            # Store advantages and returns
            idx = indices[0]
            length = len(indices)
            
            # Handle wrap-around if needed
            if idx + length > self.max_size:
                # Split into two parts
                first_part_len = self.max_size - idx
                
                # First part (from idx to end of buffer)
                self.advantages[idx:, agent_idx] = advantages[:first_part_len]
                self.returns[idx:, agent_idx] = returns[:first_part_len]
                
                # Second part (from start of buffer)
                second_part_len = length - first_part_len
                self.advantages[:second_part_len, agent_idx] = advantages[first_part_len:]
                self.returns[:second_part_len, agent_idx] = returns[first_part_len:]
            else:
                # No wrap-around
                self.advantages[idx:idx+length, agent_idx] = advantages
                self.returns[idx:idx+length, agent_idx] = returns
    
    def _will_overlap(self, start1: int, end1: int, start2: int, end2: int) -> bool:
        """
        Check if two ranges will overlap, considering buffer wraparound.
        
        Args:
            start1, end1: First range
            start2, end2: Second range
            
        Returns:
            True if ranges overlap, False otherwise
        """
        # Convert to regular ranges by unwrapping
        if end1 < start1:  # First range wraps around
            end1 += self.max_size
        if end2 < start2:  # Second range wraps around
            end2 += self.max_size
            
        # Check for overlap
        return (start1 <= start2 <= end1) or (start2 <= start1 <= end2)
    
    def sample_buffer(self):
        """
        Sample a batch from the buffer for learning.
        
        Returns:
            Dictionary with batch data
        """
        if not self.ready_to_learn or sum(self.episode_lengths) < self.batch_size:
            return None
            
        # Get valid indices (from completed episodes only)
        valid_indices = []
        for start, end in self.episode_borders:
            if start <= end:
                valid_indices.extend(range(start, end + 1))
            else:
                # Handle wrap-around
                valid_indices.extend(range(start, self.max_size))
                valid_indices.extend(range(0, end + 1))
        
        # Make sure we have enough valid transitions
        if len(valid_indices) < self.batch_size:
            return None
            
        # Randomly sample batch_size indices
        batch_indices = np.random.choice(valid_indices, self.batch_size, replace=False)
        
        # Create batch
        batch = {
            'states': self.states[batch_indices],
            'actions': self.actions[batch_indices],
            'rewards': self.rewards[batch_indices],
            'next_states': self.next_states[batch_indices],
            'dones': self.dones[batch_indices],
            'values': self.values[batch_indices],
            'log_probs': self.log_probs[batch_indices],
            'advantages': self.advantages[batch_indices],
            'returns': self.returns[batch_indices],
            'individual_obs': [self.observations[i][batch_indices] for i in range(self.n_agents)],
            'individual_next_obs': [self.next_observations[i][batch_indices] for i in range(self.n_agents)]
        }
        
        return batch
        
    def ready(self) -> bool:
        """
        Check if buffer has enough data for learning.
        
        Returns:
            True if buffer is ready for learning, False otherwise
        """
        return self.ready_to_learn and sum(self.episode_lengths) >= self.batch_size
    
    def clear(self):
        """Clear the buffer entirely."""
        self.ptr = 0
        self.size = 0
        self.episode_start = 0
        self.current_episode_length = 0
        self.episode_lengths = []
        self.episode_borders = []
        self.episodes_in_buffer = 0
        self.ready_to_learn = False 