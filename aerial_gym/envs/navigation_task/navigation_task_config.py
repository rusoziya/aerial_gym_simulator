from aerial_gym.config.task_config.navigation_task_config import task_config as base_task_config

class NavigationTaskConfig:
    """
    Configuration wrapper for the navigation task
    """
    
    def __init__(self):
        # Create task configuration dictionary
        self.task_config = {
            "seed": base_task_config.seed,
            "sim_name": base_task_config.sim_name,
            "env_name": base_task_config.env_name,
            "robot_name": base_task_config.robot_name,
            "controller_name": base_task_config.controller_name,
            "args": base_task_config.args,
            "num_envs": base_task_config.num_envs,
            "use_warp": base_task_config.use_warp,
            "headless": base_task_config.headless,
            "device": base_task_config.device,
            "observation_space_dim": base_task_config.observation_space_dim,
            "privileged_observation_space_dim": base_task_config.privileged_observation_space_dim,
            "action_space_dim": base_task_config.action_space_dim,
            "episode_len_steps": base_task_config.episode_len_steps,
            "return_state_before_reset": base_task_config.return_state_before_reset,
            "target_min_ratio": base_task_config.target_min_ratio,
            "target_max_ratio": base_task_config.target_max_ratio,
            "reward_parameters": base_task_config.reward_parameters,
            "action_transformation_function": base_task_config.action_transformation_function,
        }
        
        # Create environment configuration dictionary
        self.env_config = {
            "num_envs": base_task_config.num_envs,
        }
        
        # Create simulation configuration dictionary  
        self.sim_config = {
            "sim_name": base_task_config.sim_name,
            "env_name": base_task_config.env_name,
            "use_warp": base_task_config.use_warp,
            "headless": base_task_config.headless,
            "device": base_task_config.device,
        }
        
        # Store class references for advanced configuration
        self.vae_config = base_task_config.vae_config
        self.curriculum = base_task_config.curriculum 