import torch
import numpy as np
from aerial_gym.task.base_task import BaseTask
from aerial_gym.utils.math import *

class MultiAgentBaseTask(BaseTask):
    """
    Base class for multi-agent tasks extending the single-agent BaseTask
    """
    
    def __init__(self, task_config):
        super().__init__(task_config)
        
        # Extract multi-agent configurations from task_config
        self.num_robots_per_env = getattr(task_config, 'num_robots_per_env', 1)
        self.enable_inter_agent_obs = getattr(task_config, 'enable_inter_agent_obs', False)
        self.communication_range = getattr(task_config, 'communication_range', 10.0)  # meters
        
        # Multi-agent specific configurations
        self.shared_reward = getattr(task_config, 'shared_reward', False)
        self.individual_rewards = getattr(task_config, 'individual_rewards', True)
        
        # Device will be set from task_config
        self.device = getattr(task_config, 'device', 'cuda')
        self.num_envs = getattr(task_config, 'num_envs', 16)
        
        # Initialize multi-agent observation and action spaces
        self._setup_multi_agent_spaces()
        
        # Initialize multi-agent specific attributes
        self.robot_positions = torch.zeros(
            self.num_envs, self.num_robots_per_env, 3, device=self.device
        )
        self.robot_velocities = torch.zeros(
            self.num_envs, self.num_robots_per_env, 3, device=self.device
        )
        
    def _setup_multi_agent_spaces(self):
        """Setup observation and action spaces for multi-agent scenario"""
        # Check if single-agent spaces are already defined
        if hasattr(self, 'observation_space') and self.observation_space is not None:
            single_obs_dim = self.observation_space.shape[0]
        else:
            single_obs_dim = 10  # Default observation dimension
            
        if hasattr(self, 'action_space') and self.action_space is not None:
            single_action_dim = self.action_space.shape[0]
        else:
            single_action_dim = 4  # Default action dimension
        
        # Base observation for each robot
        base_obs_dim = single_obs_dim
        
        # Add inter-agent observations if enabled
        if self.enable_inter_agent_obs:
            # Add relative positions, velocities of other agents
            inter_agent_obs_dim = (self.num_robots_per_env - 1) * 6  # pos + vel for other agents
            base_obs_dim += inter_agent_obs_dim
            
        # Multi-agent observation space: (num_robots_per_env, obs_dim)
        try:
            import gymnasium as gym
        except ImportError:
            import gym
            
        self.multi_agent_observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.num_robots_per_env, base_obs_dim),
            dtype=np.float32
        )
        
        # Multi-agent action space: (num_robots_per_env, action_dim)  
        if hasattr(self, 'action_space') and self.action_space is not None:
            self.multi_agent_action_space = gym.spaces.Box(
                low=self.action_space.low,
                high=self.action_space.high,
                shape=(self.num_robots_per_env, single_action_dim),
                dtype=np.float32
            )
        else:
            self.multi_agent_action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_robots_per_env, single_action_dim),
                dtype=np.float32
            )
        
    def get_multi_agent_observations(self):
        """
        Get observations for all robots in all environments
        Returns: tensor of shape (num_envs, num_robots_per_env, obs_dim)
        """
        # Get observations from environment
        if hasattr(self, 'env_manager') and self.env_manager is not None:
            # Use the standard environment get_obs method
            obs_dict = self.env_manager.get_obs()
            
            # Extract robot state information
            robot_state = obs_dict.get("robot_state", torch.zeros(self.num_envs, 13, device=self.device))
            
            # For multi-agent setup, we simulate multiple agents by using multiple environments
            # Each environment represents one agent
            if self.num_robots_per_env == 1:
                # Single robot per environment - reshape to multi-agent format
                processed_obs = robot_state.unsqueeze(1)  # (num_envs, 1, obs_dim)
            else:
                # Multiple robots per environment - need to handle differently
                # For now, duplicate the single robot observation for each agent
                processed_obs = robot_state.unsqueeze(1).expand(-1, self.num_robots_per_env, -1)
                
            return processed_obs
        else:
            # Fallback: return zeros if no environment manager
            obs_dim = getattr(self, 'observation_space', gym.spaces.Box(low=-1, high=1, shape=(13,))).shape[0]
            return torch.zeros(self.num_envs, self.num_robots_per_env, obs_dim, device=self.device)
        
    def _get_inter_agent_observations(self, robot_idx):
        """
        Get relative positions and velocities of other robots within communication range
        """
        current_robot_pos = self.robot_positions[:, robot_idx, :]  # (num_envs, 3)
        current_robot_vel = self.robot_velocities[:, robot_idx, :]  # (num_envs, 3)
        
        inter_agent_obs_list = []
        
        for other_robot_idx in range(self.num_robots_per_env):
            if other_robot_idx == robot_idx:
                continue
                
            other_robot_pos = self.robot_positions[:, other_robot_idx, :]
            other_robot_vel = self.robot_velocities[:, other_robot_idx, :]
            
            # Calculate relative position and velocity
            rel_pos = other_robot_pos - current_robot_pos
            rel_vel = other_robot_vel - current_robot_vel
            
            # Calculate distance
            distance = torch.norm(rel_pos, dim=-1, keepdim=True)
            
            # Create communication mask (robots outside range get zero observations)
            comm_mask = (distance <= self.communication_range).float()
            
            # Create observation vector: [rel_pos_x, rel_pos_y, rel_pos_z, rel_vel_x, rel_vel_y, rel_vel_z, distance]
            inter_obs = torch.cat([rel_pos, rel_vel, distance], dim=-1)
            inter_obs = inter_obs * comm_mask  # Zero out observations for robots outside communication range
            
            inter_agent_obs_list.append(inter_obs)
            
        if inter_agent_obs_list:
            return torch.cat(inter_agent_obs_list, dim=-1)
        else:
            # No other robots, return zero observations
            return torch.zeros(
                self.num_envs, 
                (self.num_robots_per_env - 1) * 7,  # 7 = rel_pos(3) + rel_vel(3) + distance(1)
                device=self.device
            )
            
    def check_inter_robot_collisions(self):
        """
        Check for collisions between robots in the same environment
        Returns: boolean tensor indicating collision status
        """
        collision_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        min_distance = 0.5  # Minimum allowed distance between robots
        
        for i in range(self.num_robots_per_env):
            for j in range(i + 1, self.num_robots_per_env):
                pos_i = self.robot_positions[:, i, :]
                pos_j = self.robot_positions[:, j, :]
                distance = torch.norm(pos_i - pos_j, dim=-1)
                collision_mask = collision_mask | (distance < min_distance)
                
        return collision_mask
        
    def compute_formation_rewards(self, desired_formation, formation_scale=1.0):
        """
        Compute rewards based on maintaining a desired formation
        
        Args:
            desired_formation: tensor of shape (num_robots_per_env, 3) with relative positions
            formation_scale: scalar to scale the formation size
        """
        if self.num_robots_per_env < 2:
            return torch.zeros(self.num_envs, device=self.device)
            
        formation_rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Scale the desired formation
        scaled_formation = desired_formation * formation_scale
        
        # Calculate center of mass for each environment
        center_of_mass = torch.mean(self.robot_positions, dim=1)  # (num_envs, 3)
        
        # Calculate desired positions relative to center of mass
        desired_positions = center_of_mass.unsqueeze(1) + scaled_formation.unsqueeze(0)  # (num_envs, num_robots, 3)
        
        # Calculate formation error
        position_errors = torch.norm(self.robot_positions - desired_positions, dim=-1)  # (num_envs, num_robots)
        formation_error = torch.mean(position_errors, dim=1)  # (num_envs,)
        
        # Convert to reward (higher reward for smaller error)
        formation_rewards = torch.exp(-formation_error)
        
        return formation_rewards
        
    def compute_multi_agent_rewards(self, actions):
        """
        Compute rewards for multi-agent scenario
        Returns: dict with individual and optionally shared rewards
        """
        rewards = {}
        
        if self.individual_rewards:
            individual_rewards = torch.zeros(
                self.num_envs, self.num_robots_per_env, device=self.device
            )
            
            for robot_idx in range(self.num_robots_per_env):
                robot_reward = self._compute_individual_reward(robot_idx, actions)
                individual_rewards[:, robot_idx] = robot_reward
                
            rewards["individual"] = individual_rewards
            
        if self.shared_reward:
            shared_reward = self._compute_shared_reward(actions)
            rewards["shared"] = shared_reward
            
        return rewards
        
    def _compute_individual_reward(self, robot_idx, actions):
        """Override in specific tasks"""
        return torch.zeros(self.num_envs, device=self.device)
        
    def _compute_shared_reward(self, actions):
        """Override in specific tasks"""
        return torch.zeros(self.num_envs, device=self.device)
        
    def _process_robot_observations(self, robot_obs, robot_idx):
        """Process observations for a specific robot - override in specific tasks"""
        return robot_obs["robot_state"]  # Default implementation
        
    def reset(self):
        """Reset multi-agent environment"""
        super().reset()
        return self.get_multi_agent_observations()
        
    def step(self, actions):
        """
        Step function for multi-agent environment
        actions: tensor of shape (num_envs, num_robots_per_env, action_dim)
        """
        # Step the environment with multi-agent actions
        self.env_manager.step(actions)
        
        # Get multi-agent observations
        observations = self.get_multi_agent_observations()
        
        # Compute rewards
        rewards = self.compute_multi_agent_rewards(actions)
        
        # Check termination conditions
        dones = self._check_multi_agent_termination()
        
        # Prepare info dict
        info = self._get_multi_agent_info()
        
        return observations, rewards, dones, info
        
    def _check_multi_agent_termination(self):
        """Check termination conditions for multi-agent scenario"""
        # Default: environment terminates if any robot violates termination conditions
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for robot_idx in range(self.num_robots_per_env):
            robot_dones = self._check_individual_termination(robot_idx)
            dones = dones | robot_dones
            
        return dones
        
    def _check_individual_termination(self, robot_idx):
        """Check termination for individual robot - override in specific tasks"""
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
    def _get_multi_agent_info(self):
        """Get info dict for multi-agent scenario"""
        return {} 