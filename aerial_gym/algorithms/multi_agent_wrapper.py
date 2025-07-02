import torch
import numpy as np
import gymnasium as gym
from typing import Dict, Any, List, Tuple

class MultiAgentAerialGymWrapper:
    """
    Wrapper to make Aerial Gym compatible with multi-agent RL libraries
    """
    
    def __init__(self, task, agent_ids=None):
        self.task = task
        self.num_agents = task.num_robots_per_env
        
        # Generate agent IDs
        if agent_ids is None:
            self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        else:
            self.agent_ids = agent_ids
            
        # Create observation and action spaces for each agent
        self._setup_spaces()
        
    def _setup_spaces(self):
        """Setup observation and action spaces compatible with MARL libraries"""
        # Individual observation space for each agent
        single_obs_shape = self.task.multi_agent_observation_space.shape[1]
        single_action_shape = self.task.multi_agent_action_space.shape[1]
        
        self.observation_spaces = {
            agent_id: gym.spaces.Box(
                low=-np.inf,
                high=np.inf, 
                shape=(single_obs_shape,),
                dtype=np.float32
            )
            for agent_id in self.agent_ids
        }
        
        self.action_spaces = {
            agent_id: gym.spaces.Box(
                low=self.task.multi_agent_action_space.low[0],
                high=self.task.multi_agent_action_space.high[0],
                shape=(single_action_shape,),
                dtype=np.float32
            )
            for agent_id in self.agent_ids
        }
        
    def reset(self, **kwargs):
        """Reset environment and return observations for all agents"""
        observations = self.task.reset()
        
        # Convert to agent-indexed dictionary
        agent_observations = {}
        for env_idx in range(self.task.num_envs):
            for agent_idx, agent_id in enumerate(self.agent_ids):
                key = f"{agent_id}_env_{env_idx}"
                agent_observations[key] = observations[env_idx, agent_idx].cpu().numpy()
                
        return agent_observations
        
    def step(self, actions_dict):
        """
        Step environment with actions from all agents
        
        Args:
            actions_dict: Dict mapping agent_id_env_idx to action arrays
            
        Returns:
            observations, rewards, dones, truncated, infos
        """
        # Convert action dict to tensor format expected by task
        actions_tensor = torch.zeros(
            self.task.num_envs,
            self.num_agents,
            self.task.multi_agent_action_space.shape[1],
            device=self.task.device
        )
        
        for env_idx in range(self.task.num_envs):
            for agent_idx, agent_id in enumerate(self.agent_ids):
                key = f"{agent_id}_env_{env_idx}"
                if key in actions_dict:
                    action = torch.tensor(actions_dict[key], device=self.task.device)
                    actions_tensor[env_idx, agent_idx] = action
                    
        # Step the environment
        observations, rewards, dones, info = self.task.step(actions_tensor)
        
        # Convert outputs to agent-indexed format
        agent_observations = {}
        agent_rewards = {}
        agent_dones = {}
        agent_truncated = {}
        agent_infos = {}
        
        for env_idx in range(self.task.num_envs):
            for agent_idx, agent_id in enumerate(self.agent_ids):
                key = f"{agent_id}_env_{env_idx}"
                
                # Observations
                agent_observations[key] = observations[env_idx, agent_idx].cpu().numpy()
                
                # Rewards
                if isinstance(rewards, dict):
                    if "individual" in rewards:
                        agent_rewards[key] = rewards["individual"][env_idx, agent_idx].item()
                    if "shared" in rewards:
                        # Add shared reward to individual reward
                        shared_reward = rewards["shared"][env_idx].item()
                        agent_rewards[key] = agent_rewards.get(key, 0) + shared_reward
                else:
                    agent_rewards[key] = rewards[env_idx, agent_idx].item()
                    
                # Dones
                agent_dones[key] = dones[env_idx].item()
                agent_truncated[key] = False  # No truncation in this implementation
                
                # Info
                agent_infos[key] = info
                
        return agent_observations, agent_rewards, agent_dones, agent_truncated, agent_infos

class RLLibMultiAgentWrapper(MultiAgentAerialGymWrapper):
    """
    Wrapper specifically for RLLib multi-agent environments
    """
    
    def __init__(self, task, agent_ids=None):
        super().__init__(task, agent_ids)
        
    def observation_space(self, agent_id):
        """Get observation space for specific agent"""
        return self.observation_spaces[agent_id]
        
    def action_space(self, agent_id):
        """Get action space for specific agent"""
        return self.action_spaces[agent_id]
        
    def get_agent_ids(self):
        """Get list of all agent IDs"""
        agent_ids = []
        for env_idx in range(self.task.num_envs):
            for agent_id in self.agent_ids:
                agent_ids.append(f"{agent_id}_env_{env_idx}")
        return agent_ids

class MAPPOWrapper(MultiAgentAerialGymWrapper):
    """
    Wrapper for MAPPO (Multi-Agent PPO) algorithm
    """
    
    def __init__(self, task, agent_ids=None):
        super().__init__(task, agent_ids)
        
    def get_global_state(self):
        """Get global state for centralized critic"""
        # Collect all robot positions and velocities
        robot_observations = self.task.env_manager.get_observations()
        
        global_state = []
        for robot_idx in range(self.num_agents):
            robot_obs = robot_observations[f"robot_{robot_idx}"]
            pos = robot_obs["robot_position"]  # (num_envs, 3)
            vel = robot_obs["robot_velocity"]  # (num_envs, 3)
            global_state.append(torch.cat([pos, vel], dim=1))
            
        # Concatenate all robot states
        global_state = torch.cat(global_state, dim=1)  # (num_envs, num_agents * 6)
        
        return global_state.cpu().numpy()
        
    def get_state_size(self):
        """Get size of global state for centralized critic"""
        return self.num_agents * 6  # pos (3) + vel (3) per agent

class QMIXWrapper(MultiAgentAerialGymWrapper):
    """
    Wrapper for QMIX algorithm
    """
    
    def __init__(self, task, agent_ids=None):
        super().__init__(task, agent_ids)
        
        # Convert to discrete action space for QMIX
        self._discretize_action_space()
        
    def _discretize_action_space(self):
        """Convert continuous actions to discrete for QMIX"""
        # Define discrete action grid
        self.action_discretization = 5  # 5 levels per action dimension
        
        # Create discrete action space
        action_dim = self.task.multi_agent_action_space.shape[1]
        discrete_actions = self.action_discretization ** action_dim
        
        self.action_spaces = {
            agent_id: gym.spaces.Discrete(discrete_actions)
            for agent_id in self.agent_ids
        }
        
        # Create mapping from discrete to continuous actions
        self._create_action_mapping(action_dim)
        
    def _create_action_mapping(self, action_dim):
        """Create mapping from discrete action indices to continuous actions"""
        self.discrete_to_continuous = {}
        
        # Generate all combinations of discretized actions
        levels = np.linspace(-1, 1, self.action_discretization)
        
        action_idx = 0
        for combo in np.ndindex(*[self.action_discretization] * action_dim):
            continuous_action = np.array([levels[i] for i in combo])
            self.discrete_to_continuous[action_idx] = continuous_action
            action_idx += 1
            
    def step(self, actions_dict):
        """Step with discrete actions converted to continuous"""
        # Convert discrete actions to continuous
        continuous_actions_dict = {}
        
        for agent_env_key, discrete_action in actions_dict.items():
            continuous_action = self.discrete_to_continuous[discrete_action]
            continuous_actions_dict[agent_env_key] = continuous_action
            
        # Use parent step method with continuous actions
        return super().step(continuous_actions_dict)

def create_multi_agent_env(task_config_class, algorithm="ppo", **kwargs):
    """
    Factory function to create multi-agent environment with appropriate wrapper
    
    Args:
        task_config_class: Configuration class for the task
        algorithm: RL algorithm to use ("ppo", "mappo", "qmix", "rllib")
        **kwargs: Additional arguments for task creation
                 - device: Device to use ("cuda", "cpu", "auto")
                 - headless: Whether to run without visualization (bool)
                 - num_envs: Number of parallel environments (int)
        
    Returns:
        Wrapped multi-agent environment
    """
    from aerial_gym.envs.navigation_task.multi_agent_navigation_task import MultiAgentNavigationTask
    
    # Create configuration
    config = task_config_class()
    
    # Update device in config if provided
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(config, 'task_config') and isinstance(config.task_config, dict):
        config.task_config['device'] = device
        
        # Handle headless parameter
        if "headless" in kwargs:
            config.task_config["headless"] = kwargs["headless"]
            if hasattr(config, 'sim_config') and isinstance(config.sim_config, dict):
                config.sim_config["headless"] = kwargs["headless"]
        
        # Handle num_envs parameter
        if "num_envs" in kwargs:
            config.task_config["num_envs"] = kwargs["num_envs"]
            config.env_config["num_envs"] = kwargs["num_envs"]
        
        # Convert task_config dict to object with attributes that BaseTask expects
        class TaskConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
        
        task_config_obj = TaskConfig(config.task_config)
    else:
        # Fallback: try to set device on config directly
        if hasattr(config, '__dict__'):
            config.device = device
            
            # Handle headless parameter
            if "headless" in kwargs:
                config.headless = kwargs["headless"]
            
            # Handle num_envs parameter
            if "num_envs" in kwargs:
                config.num_envs = kwargs["num_envs"]
                
        task_config_obj = config
    
    # Create task with the properly formatted task config
    task = MultiAgentNavigationTask(task_config_obj)
    
    # Apply appropriate wrapper
    if algorithm.lower() == "rllib":
        return RLLibMultiAgentWrapper(task)
    elif algorithm.lower() == "mappo":
        return MAPPOWrapper(task)
    elif algorithm.lower() == "qmix":
        return QMIXWrapper(task)
    else:
        return MultiAgentAerialGymWrapper(task)

# Example usage patterns for different algorithms
def get_rllib_config():
    """Example RLLib configuration"""
    return {
        "multiagent": {
            "policies": {
                "shared_policy": (None, None, None, {})
            },
            "policy_mapping_fn": lambda agent_id: "shared_policy",
        },
        "framework": "torch",
        "num_workers": 4,
        "train_batch_size": 4000,
    }

def get_mappo_config():
    """Example MAPPO configuration"""
    return {
        "use_centralized_critic": True,
        "use_global_state": True,
        "batch_size": 32,
        "lr": 3e-4,
        "eps": 1e-5,
        "entropy_coef": 0.01,
    }

def get_qmix_config():
    """Example QMIX configuration"""
    return {
        "mixer": "qmix",
        "mixing_embed_dim": 32,
        "hypernet_layers": 2,
        "hypernet_embed": 64,
        "lr": 5e-4,
        "eps": 0.05,
    } 