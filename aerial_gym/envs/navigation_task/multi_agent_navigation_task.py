import torch
import numpy as np
from aerial_gym.envs.base.multi_agent_base_task import MultiAgentBaseTask
from aerial_gym.utils.math import *
from aerial_gym.sim.sim_builder import SimBuilder

class MultiAgentNavigationTask(MultiAgentBaseTask):
    """
    Multi-agent navigation task where multiple drones cooperatively navigate to targets
    Supports the hover recipe with proper reward calculation
    """
    
    def __init__(self, task_config):
        # Set up multi-agent specific attributes before calling super().__init__
        self.num_robots_per_env = getattr(task_config, "num_robots_per_env", 3)
        
        super().__init__(task_config)
        
        # Episode step tracking
        self.episode_len_steps = getattr(task_config, "episode_len_steps", 500)
        self.current_episode_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        
        # Initialize simulation environment using SimBuilder
        self.sim_env = SimBuilder().build_env(
            sim_name=getattr(task_config, "sim_name", "base_sim"),
            env_name=getattr(task_config, "env_name", "empty_env"),
            robot_name=getattr(task_config, "robot_name", "base_quadrotor"),
            controller_name=getattr(task_config, "controller_name", "lee_position_control"),
            args=getattr(task_config, "args", None),
            device=self.device,
            num_envs=self.num_envs * self.num_robots_per_env,  # Total robots across all envs
            use_warp=getattr(task_config, "use_warp", False),
            headless=getattr(task_config, "headless", True),
        )
        
        # Set env_manager reference for compatibility with base class methods
        self.env_manager = self.sim_env
        
        # Hover recipe reward parameters
        self.position_error_weight = getattr(task_config, "position_error_weight", 1.0)
        self.velocity_penalty_weight = getattr(task_config, "velocity_penalty_weight", 0.1)
        self.collision_penalty_weight = getattr(task_config, "collision_penalty_weight", 10.0)
        self.soft_collision_radius = getattr(task_config, "soft_collision_radius", 0.5)
        self.hard_collision_radius = getattr(task_config, "hard_collision_radius", 0.2)
        
        # Task parameters
        self.target_radius = getattr(task_config, "target_radius", 1.0)
        self.formation_reward_weight = getattr(task_config, "formation_reward_weight", 0.1)
        self.min_inter_robot_distance = getattr(task_config, "min_inter_robot_distance", 2.0)
        
        # Formation control
        self.enable_formation_control = getattr(task_config, "enable_formation_control", False)
        self.desired_formation = getattr(task_config, "desired_formation", "triangle")
        self.formation_scale = getattr(task_config, "formation_scale", 2.0)
        
        # Fixed target and initial positions from config
        self.target_positions_config = getattr(task_config, "target_positions", [
            [0.0, 0.0, 3.0],      # Agent 0: center
            [-1.5, -1.0, 3.0],    # Agent 1: left-back 
            [1.5, -1.0, 3.0]      # Agent 2: right-back
        ])
        
        self.init_positions_config = getattr(task_config, "init_positions", [
            [0.0, 0.0, 0.2],      # Agent 0: center-front
            [-1.5, -1.0, 0.2],    # Agent 1: left-back
            [1.5, -1.0, 0.2]      # Agent 2: right-back
        ])
        
        # Convert to tensors
        self.target_positions = torch.tensor(
            self.target_positions_config, device=self.device, dtype=torch.float32
        ).unsqueeze(0).repeat(self.num_envs, 1, 1)  # (num_envs, num_robots, 3)
        
        self.init_positions = torch.tensor(
            self.init_positions_config, device=self.device, dtype=torch.float32
        ).unsqueeze(0).repeat(self.num_envs, 1, 1)  # (num_envs, num_robots, 3)
        
        # Setup multi-agent spaces
        self._setup_multi_agent_spaces()
        
        print(f"✅ MultiAgentNavigationTask initialized:")
        print(f"   - Robots per env: {self.num_robots_per_env}")
        print(f"   - Total robots: {self.num_envs * self.num_robots_per_env}")
        print(f"   - Episode length: {self.episode_len_steps} steps")
        print(f"   - Target positions: {self.target_positions_config}")
        
    def _setup_multi_agent_spaces(self):
        """Setup observation and action spaces for multi-agent scenario with correct dimensions"""
        try:
            import gymnasium as gym
        except ImportError:
            import gym
            
        # Hover recipe observation space (19D as specified):
        # - Position & velocity relative to hover point: 6D
        # - Orientation (quaternion) & angular rates: 7D  
        # - Relative positions of other 2 drones: 6D
        obs_dim = 19
        
        # Multi-agent observation space: (num_robots_per_env, obs_dim)
        self.multi_agent_observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.num_robots_per_env, obs_dim),
            dtype=np.float32
        )
        
        # Multi-agent action space: (num_robots_per_env, action_dim) 
        # 4D continuous actions for quadrotor control
        action_dim = 4
        self.multi_agent_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_robots_per_env, action_dim),
            dtype=np.float32
        )
        
    def close(self):
        """Close the environment and clean up resources"""
        if hasattr(self, 'sim_env') and self.sim_env is not None:
            self.sim_env.delete_env()
            
    def render(self, mode="human"):
        """Render the environment"""
        if hasattr(self, 'sim_env') and self.sim_env is not None:
            return self.sim_env.render()
        return None
        
    def reset_idx(self, env_ids):
        """Reset specific environments by their IDs"""
        # Reset episode step counters for these environments
        self.current_episode_steps[env_ids] = 0
        
        # Reset the simulation for these environments FIRST
        flat_env_ids = []
        for env_idx in env_ids:
            for robot_idx in range(self.num_robots_per_env):
                flat_env_ids.append(env_idx * self.num_robots_per_env + robot_idx)
        
        if len(flat_env_ids) > 0:
            flat_env_ids_tensor = torch.tensor(flat_env_ids, device=self.device)
            self.sim_env.reset_idx(flat_env_ids_tensor)
        
        # NOW set robot positions to our desired initial positions (after simulation reset)
        if hasattr(self.sim_env, 'robot_manager') and hasattr(self.sim_env.robot_manager, 'robot'):
            for env_idx in env_ids:
                for robot_idx in range(self.num_robots_per_env):
                    # Calculate the absolute robot index in the flattened robot array
                    abs_robot_idx = env_idx * self.num_robots_per_env + robot_idx
                    
                    # Set initial position directly in robot state tensor
                    init_pos = self.init_positions[env_idx, robot_idx]
                    self.sim_env.robot_manager.robot.robot_state[abs_robot_idx, :3] = init_pos
                    
                    # Set zero velocity for stable start
                    self.sim_env.robot_manager.robot.robot_state[abs_robot_idx, 7:10] = 0.0  # linear velocity
                    self.sim_env.robot_manager.robot.robot_state[abs_robot_idx, 10:13] = 0.0  # angular velocity
            
            # Update the robot states to sync with simulation
            self.sim_env.robot_manager.robot.update_states()
        
    def reset(self):
        """Reset the multi-agent navigation environment"""
        # Reset all environments
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(all_env_ids)
        return self.get_multi_agent_observations()
        
    def get_multi_agent_observations(self):
        """
        Get observations for all robots in all environments following the hover recipe
        Returns: tensor of shape (num_envs, num_robots_per_env, obs_dim)
        """
        # Get observations from the simulation environment
        obs_dict = self.sim_env.get_obs()
        
        # Extract robot state - this contains position, velocity, orientation, etc.
        all_robot_states = obs_dict.get("robot_state_tensor", obs_dict.get("robot_state", torch.zeros(
            self.num_envs * self.num_robots_per_env, 13, device=self.device
        )))
        
        # Reshape to (num_envs, num_robots_per_env, 13)
        robot_states = all_robot_states.view(self.num_envs, self.num_robots_per_env, 13)
        
        # Build observations following the hover recipe (19D per agent)
        processed_obs = torch.zeros(self.num_envs, self.num_robots_per_env, 19, device=self.device)
        
        for env_idx in range(self.num_envs):
            for robot_idx in range(self.num_robots_per_env):
                obs_idx = 0
                
                # Current robot state
                robot_state = robot_states[env_idx, robot_idx]  # (13,)
                robot_pos = robot_state[:3]  # position
                robot_vel = robot_state[7:10]  # linear velocity (assuming indices 7-9)
                robot_quat = robot_state[3:7]  # orientation quaternion
                robot_angular_vel = robot_state[10:13]  # angular velocity
                
                # Target position for this robot
                target_pos = self.target_positions[env_idx, robot_idx]
                
                # 1. Position & velocity relative to hover point (6D)
                relative_pos = robot_pos - target_pos
                processed_obs[env_idx, robot_idx, obs_idx:obs_idx+3] = relative_pos
                obs_idx += 3
                processed_obs[env_idx, robot_idx, obs_idx:obs_idx+3] = robot_vel
                obs_idx += 3
                
                # 2. Orientation (quaternion) & angular rates (7D)
                processed_obs[env_idx, robot_idx, obs_idx:obs_idx+4] = robot_quat
                obs_idx += 4
                processed_obs[env_idx, robot_idx, obs_idx:obs_idx+3] = robot_angular_vel
                obs_idx += 3
                
                # 3. Relative positions of other 2 drones (6D)
                other_robot_count = 0
                for other_robot_idx in range(self.num_robots_per_env):
                    if other_robot_idx != robot_idx and other_robot_count < 2:
                        other_robot_pos = robot_states[env_idx, other_robot_idx, :3]
                        relative_other_pos = other_robot_pos - robot_pos
                        processed_obs[env_idx, robot_idx, obs_idx:obs_idx+3] = relative_other_pos
                        obs_idx += 3
                        other_robot_count += 1
                
                # Fill remaining slots with zeros if fewer than 2 other robots
                while other_robot_count < 2:
                    processed_obs[env_idx, robot_idx, obs_idx:obs_idx+3] = 0.0
                    obs_idx += 3
                    other_robot_count += 1
        
        return processed_obs
        
    def compute_multi_agent_rewards(self, actions):
        """
        Compute rewards following the hover recipe:
        R = -Σ||p_i - p_target||² - α*Σ||v_i||² - β*Σ exp(-||p_i - p_j||²/d₀²)
        """
        # Get robot states
        obs_dict = self.sim_env.get_obs()
        all_robot_states = obs_dict.get("robot_state_tensor", obs_dict.get("robot_state", torch.zeros(
            self.num_envs * self.num_robots_per_env, 13, device=self.device
        )))
        robot_states = all_robot_states.view(self.num_envs, self.num_robots_per_env, 13)
        
        # Initialize rewards
        rewards = torch.zeros(self.num_envs, self.num_robots_per_env, device=self.device)
        
        for env_idx in range(self.num_envs):
            env_reward = 0.0  # Shared global reward for this environment
            
            # 1. Position error term: -Σ||p_i - p_target||²
            position_error_sum = 0.0
            for robot_idx in range(self.num_robots_per_env):
                robot_pos = robot_states[env_idx, robot_idx, :3]
                target_pos = self.target_positions[env_idx, robot_idx]
                position_error = torch.norm(robot_pos - target_pos, dim=0) ** 2
                position_error_sum += position_error
            
            env_reward -= self.position_error_weight * position_error_sum
            
            # 2. Velocity penalty: -α*Σ||v_i||²
            velocity_penalty_sum = 0.0
            for robot_idx in range(self.num_robots_per_env):
                robot_vel = robot_states[env_idx, robot_idx, 7:10]
                velocity_magnitude_sq = torch.norm(robot_vel, dim=0) ** 2
                velocity_penalty_sum += velocity_magnitude_sq
            
            env_reward -= self.velocity_penalty_weight * velocity_penalty_sum
            
            # 3. Collision penalty: -β*Σ exp(-||p_i - p_j||²/d₀²)
            collision_penalty_sum = 0.0
            for i in range(self.num_robots_per_env):
                for j in range(i + 1, self.num_robots_per_env):
                    pos_i = robot_states[env_idx, i, :3]
                    pos_j = robot_states[env_idx, j, :3]
                    distance_sq = torch.norm(pos_i - pos_j, dim=0) ** 2
                    collision_term = torch.exp(-distance_sq / (self.soft_collision_radius ** 2))
                    collision_penalty_sum += collision_term
            
            env_reward -= self.collision_penalty_weight * collision_penalty_sum
            
            # Apply shared reward to all agents in this environment
            for robot_idx in range(self.num_robots_per_env):
                rewards[env_idx, robot_idx] = env_reward
        
        return rewards
        
    def _check_multi_agent_termination(self):
        """Check termination conditions for multi-agent setup"""
        # Get robot states
        obs_dict = self.sim_env.get_obs()
        all_robot_states = obs_dict.get("robot_state_tensor", obs_dict.get("robot_state", torch.zeros(
            self.num_envs * self.num_robots_per_env, 13, device=self.device
        )))
        robot_states = all_robot_states.view(self.num_envs, self.num_robots_per_env, 13)
        
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for env_idx in range(self.num_envs):
            env_done = False
            
            # Check episode length first
            if self.current_episode_steps[env_idx] >= self.episode_len_steps:
                print(f"   DEBUG: Environment {env_idx} terminated due to episode length: {self.current_episode_steps[env_idx]} >= {self.episode_len_steps}")
                env_done = True
            else:
                # Check each robot in this environment
                for robot_idx in range(self.num_robots_per_env):
                    robot_pos = robot_states[env_idx, robot_idx, :3]
                    
                    # Check bounds (very generous to allow learning)
                    out_of_bounds = (
                        (robot_pos[0] < -50) | (robot_pos[0] > 50) |
                        (robot_pos[1] < -50) | (robot_pos[1] > 50) |
                        (robot_pos[2] < -5) | (robot_pos[2] > 20)  # Very permissive height bounds for debugging
                    )
                    
                    if out_of_bounds:
                        print(f"   DEBUG: Environment {env_idx}, Robot {robot_idx} out of bounds: pos={robot_pos}")
                        env_done = True
                        break
                    
                    # Check hard collisions with other robots (more relaxed)
                    for other_idx in range(self.num_robots_per_env):
                        if other_idx != robot_idx:
                            other_pos = robot_states[env_idx, other_idx, :3]
                            distance = torch.norm(robot_pos - other_pos, dim=0)
                            if distance < self.hard_collision_radius:
                                print(f"   DEBUG: Environment {env_idx}, Robot {robot_idx} collision with robot {other_idx}: distance={distance:.3f} < {self.hard_collision_radius}")
                                env_done = True
                                break
                    
                    if env_done:
                        break
            
            dones[env_idx] = env_done
            if env_done:
                print(f"   DEBUG: Environment {env_idx} marked as done")
        
        print(f"   DEBUG: Final custom termination result: {dones}")
        return dones
        
    def step(self, actions):
        """
        Step function for multi-agent environment
        actions: tensor of shape (num_envs, num_robots_per_env, action_dim)
        """
        # Increment episode step counters
        self.current_episode_steps += 1
        
        # Flatten actions for the simulation environment
        # actions: (num_envs, num_robots_per_env, action_dim) -> (total_robots, action_dim)
        flat_actions = actions.view(-1, actions.shape[-1])
        
        # Step the simulation environment
        self.sim_env.step(flat_actions)
        
        # Get observation dictionary from simulation
        obs_dict = self.sim_env.get_obs()
        
        # CRITICAL: Check simulation-detected crashes (following standard pattern)
        # All other tasks call compute_rewards_and_crashes() to get simulation crashes
        sim_rewards, sim_crashes = self.compute_simulation_crashes(obs_dict)
        
        # Get multi-agent observations
        observations = self.get_multi_agent_observations()
        
        # Compute our custom multi-agent rewards 
        rewards = self.compute_multi_agent_rewards(actions)
        
        # Check episode length truncations (following standard pattern)
        truncations = torch.where(
            self.current_episode_steps >= self.episode_len_steps,
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        
        # Check our custom multi-agent termination conditions
        custom_terminations = self._check_multi_agent_termination()
        
        # Combine simulation crashes with our custom terminations and truncations
        # Environment is done if: simulation crash OR custom termination OR truncation
        dones = sim_crashes | custom_terminations | truncations
        
        # Prepare info dict
        info = self._get_multi_agent_info()
        info["sim_crashes"] = sim_crashes
        info["custom_terminations"] = custom_terminations 
        info["truncations"] = truncations
        
        return observations, rewards, dones, info
    
    def compute_simulation_crashes(self, obs_dict):
        """
        Check for simulation-level crashes using the standard pattern.
        This follows the same approach as other tasks to detect physics instabilities,
        extreme velocities, etc. that the simulation engine detects.
        """
        # Use the crashes detected by the simulation
        sim_crashes = obs_dict.get("crashes", torch.zeros(
            self.num_envs * self.num_robots_per_env, dtype=torch.bool, device=self.device
        ))
        
        # Reshape to per-environment (any robot crash = environment crash)
        if len(sim_crashes.shape) == 1:
            sim_crashes = sim_crashes.view(self.num_envs, self.num_robots_per_env)
        
        # Environment crashes if ANY robot in that environment crashes
        env_crashes = torch.any(sim_crashes, dim=1)
        
        # For rewards, return zero (we compute rewards separately)
        sim_rewards = torch.zeros(self.num_envs, self.num_robots_per_env, device=self.device)
        
        return sim_rewards, env_crashes
        
    def _get_multi_agent_info(self):
        """Get info dict for multi-agent navigation"""
        obs_dict = self.sim_env.get_obs()
        all_robot_states = obs_dict.get("robot_state_tensor", obs_dict.get("robot_state", torch.zeros(
            self.num_envs * self.num_robots_per_env, 13, device=self.device
        )))
        robot_states = all_robot_states.view(self.num_envs, self.num_robots_per_env, 13)
        
        info = {
            "target_distances": [],
            "formation_error": None,
            "collision_count": 0
        }
        
        # Calculate distances to targets
        for robot_idx in range(self.num_robots_per_env):
            distances = []
            for env_idx in range(self.num_envs):
                robot_pos = robot_states[env_idx, robot_idx, :3]
                target_pos = self.target_positions[env_idx, robot_idx]
                distance = torch.norm(target_pos - robot_pos, dim=0)
                distances.append(distance.item())
            info["target_distances"].append(np.mean(distances))
            
        return info 