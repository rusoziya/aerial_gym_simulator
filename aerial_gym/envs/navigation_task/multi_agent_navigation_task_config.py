import numpy as np
from aerial_gym.envs.navigation_task.navigation_task_config import NavigationTaskConfig

class MultiAgentNavigationTaskConfig(NavigationTaskConfig):
    """
    Configuration for multi-agent navigation task
    """
    
    def __init__(self):
        super().__init__()
        
        # Multi-agent specific configurations
        self.task_config["num_robots_per_env"] = 3  # Number of robots per environment
        self.task_config["enable_inter_agent_obs"] = True  # Enable inter-agent observations
        self.task_config["communication_range"] = 15.0  # Communication range in meters
        
        # Reward structure
        self.task_config["shared_reward"] = True  # Enable shared team rewards
        self.task_config["individual_rewards"] = True  # Enable individual rewards
        
        # Navigation parameters
        self.task_config["target_radius"] = 1.5
        self.task_config["formation_reward_weight"] = 0.2
        self.task_config["collision_penalty"] = -5.0
        self.task_config["min_inter_robot_distance"] = 3.0
        
        # Formation control
        self.task_config["enable_formation_control"] = True
        self.task_config["desired_formation"] = "triangle"  # line, circle, triangle
        self.task_config["formation_scale"] = 4.0
        
        # Environment configuration
        self.env_config["num_robots_per_env"] = self.task_config["num_robots_per_env"]
        
        # Update observation and action dimensions
        self._update_spaces_for_multi_agent()
        
    def _update_spaces_for_multi_agent(self):
        """Update observation and action spaces for multi-agent setup"""
        # Base single-agent observation dimension
        base_obs_dim = self.task_config["observation_space_dim"]
        
        # Add relative target position (3D)
        base_obs_dim += 3
        
        # Add inter-agent observations if enabled
        if self.task_config["enable_inter_agent_obs"]:
            num_other_agents = self.task_config["num_robots_per_env"] - 1
            inter_agent_obs_dim = num_other_agents * 6  # rel_pos (3) + rel_vel (3) per agent
            base_obs_dim += inter_agent_obs_dim
            
        # Update observation space
        self.task_config["observation_space_dim"] = base_obs_dim
        
        # Action space remains the same per robot
        # Multi-agent wrapper will handle the additional dimension
        
class MultiAgentFormationTaskConfig(MultiAgentNavigationTaskConfig):
    """
    Configuration for formation flying task
    """
    
    def __init__(self):
        super().__init__()
        
        # Emphasize formation control
        self.task_config["enable_formation_control"] = True
        self.task_config["formation_reward_weight"] = 0.5
        self.task_config["shared_reward"] = True
        
        # Tighter formation
        self.task_config["formation_scale"] = 3.0
        self.task_config["min_inter_robot_distance"] = 2.0
        
        # Enable visualization - disable headless mode
        self.task_config["headless"] = False
        self.sim_config["headless"] = False
        
class MultiAgentFormationTaskConfigLight(MultiAgentNavigationTaskConfig):
    """
    Lightweight configuration for formation flying task - optimized for visualization
    """
    
    def __init__(self):
        super().__init__()
        
        # Lightweight settings for better visualization
        self.task_config["num_envs"] = 4  # Much fewer parallel environments
        self.env_config["num_envs"] = 4
        
        # Small formation with 3 agents
        self.task_config["num_robots_per_env"] = 3
        self.env_config["num_robots_per_env"] = 3
        
        # Formation control settings
        self.task_config["enable_formation_control"] = True
        self.task_config["formation_reward_weight"] = 0.5
        self.task_config["shared_reward"] = True
        
        # Compact formation for easy observation
        self.task_config["desired_formation"] = "triangle"
        self.task_config["formation_scale"] = 2.5  # Smaller, tighter formation
        self.task_config["min_inter_robot_distance"] = 1.5  # Closer together but safe
        
        # Better inter-agent observation for coordination
        self.task_config["enable_inter_agent_obs"] = True
        self.task_config["communication_range"] = 10.0
        
        # Visualization settings
        self.task_config["headless"] = False
        self.sim_config["headless"] = False
        
        # Longer episodes for better observation
        self.task_config["episode_len_steps"] = 500
        
        # Gentler rewards for smoother behavior
        self.task_config["target_radius"] = 2.0  # Larger target radius
        self.task_config["collision_penalty"] = -2.0  # Less harsh collision penalty
        
        # Re-update spaces for the new configuration
        self._update_spaces_for_multi_agent()

class MultiAgentSwarmTaskConfig(MultiAgentNavigationTaskConfig):
    """
    Configuration for large swarm tasks
    """
    
    def __init__(self):
        super().__init__()
        
        # Larger swarm
        self.task_config["num_robots_per_env"] = 6
        self.env_config["num_robots_per_env"] = 6
        
        # Looser formation for swarm
        self.task_config["desired_formation"] = "circle"
        self.task_config["formation_scale"] = 8.0
        self.task_config["min_inter_robot_distance"] = 2.5
        
        # Reduced communication range for realistic swarm
        self.task_config["communication_range"] = 10.0
        
        # Re-update spaces for larger swarm
        self._update_spaces_for_multi_agent()

class MultiAgentSurveillanceTaskConfig(MultiAgentNavigationTaskConfig):
    """
    Configuration for surveillance task with static and mobile drones
    """
    
    def __init__(self):
        super().__init__()
        
        # Mixed team: some static, some mobile
        self.task_config["num_robots_per_env"] = 4
        self.env_config["num_robots_per_env"] = 4
        
        # Surveillance specific
        self.task_config["num_static_drones"] = 1  # Number of static surveillance drones
        self.task_config["enable_camera_sharing"] = True  # Share camera feeds
        self.task_config["surveillance_coverage_reward"] = 0.3
        
        # Less emphasis on formation, more on coverage
        self.task_config["enable_formation_control"] = False
        self.task_config["formation_reward_weight"] = 0.0
        
        # Extended communication for surveillance
        self.task_config["communication_range"] = 20.0
        
        # Re-update spaces
        self._update_spaces_for_multi_agent()

class MultiAgentHoverTaskConfig:
    """
    Configuration for Multi-Agent Hover Task following the exact recipe
    """
    
    def __init__(self):
        # Required BaseTask attributes
        self.seed = 42
        
        # Environment setup
        self.num_envs = 2  # Small number for visualization
        self.num_robots_per_env = 3  # 3 drones per environment (recipe specification)
        
        # Simulation settings
        self.sim_name = "base_sim"
        self.env_name = "empty_env"
        self.robot_name = "base_quadrotor"
        self.controller_name = "lee_position_control"
        self.headless = False  # Enable visualization
        self.use_warp = False
        
        # Device settings
        self.device = "cuda"
        
        # Episode configuration
        self.episode_len_steps = 500  # Number of steps per episode
        self.return_state_before_reset = True
        
        # Task parameters
        self.target_radius = 0.3
        
        # Hover recipe reward parameters (exact values from recipe)
        self.position_error_weight = 1.0      # Main reward term
        self.velocity_penalty_weight = 0.1    # α = 0.1 (velocity penalty)
        self.collision_penalty_weight = 10.0  # β = 10 (collision penalty)
        self.soft_collision_radius = 0.5      # d₀ = 0.5m (soft collision radius)
        self.hard_collision_radius = 0.2      # R_safe = 0.2m (hard collision cutoff)
        
        # Fixed target positions for triangle formation hover (recipe specification)
        self.target_positions = [
            [0.0, 0.0, 3.0],      # Agent 0: center
            [-1.5, -1.0, 3.0],    # Agent 1: left-back 
            [1.5, -1.0, 3.0]      # Agent 2: right-back
        ]
        
        # Initial positions (start closer to targets but at safe height)
        self.init_positions = [
            [0.0, 0.0, 2.0],      # Agent 0: center - start at 2m height
            [-1.5, -1.0, 2.0],    # Agent 1: left-back - start at 2m height
            [1.5, -1.0, 2.0]      # Agent 2: right-back - start at 2m height
        ]
        
        # Formation control
        self.enable_formation_control = True
        self.desired_formation = "triangle"
        self.formation_scale = 2.0
        self.formation_reward_weight = 0.1
        
        # Episode settings
        self.episode_length_s = 10.0  # 10 seconds per episode
        self.control_frequency_hz = 50.0
        
        # Observation and action space settings
        self.observation_space_dim = 19  # Recipe-compliant: 6 + 7 + 6
        self.privileged_observation_space_dim = 19  # Same as observation space
        self.action_space_dim = 4  # 4D continuous thrust/torque
        
        # Termination conditions
        self.enable_onboard_cameras = False
        self.return_state_before_reset = True
        
        # Additional simulation args
        self.args = None
        
        # Reward parameters (required by base task)
        self.reward_parameters = {
            "position_error_weight": self.position_error_weight,
            "velocity_penalty_weight": self.velocity_penalty_weight,
            "collision_penalty_weight": self.collision_penalty_weight,
        }
        
        # Action transformation function (required by base task)
        self.action_transformation_function = None
        
        # Target ratios (required by base task)
        self.target_min_ratio = 0.5
        self.target_max_ratio = 1.5
        
        print("🚁 MultiAgentHoverTaskConfig initialized with hover recipe:")
        print(f"   - Environments: {self.num_envs}")
        print(f"   - Robots per env: {self.num_robots_per_env}")
        print(f"   - Visualization: {'ON' if not self.headless else 'OFF'}")
        print(f"   - Reward weights: α={self.velocity_penalty_weight}, β={self.collision_penalty_weight}")
        print(f"   - Target triangle: {self.target_positions}") 