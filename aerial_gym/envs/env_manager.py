class EnvManager:
    def __init__(self, env_config, sim_config, device):
        # ... existing code ...
        self.num_robots_per_env = env_config.get("num_robots_per_env", 1)
        self.robot_managers = []  # List of robot managers for multi-robot support
        
    def create_sim(self):
        # ... existing code ...
        
        # Create multiple robot managers if needed
        for robot_idx in range(self.num_robots_per_env):
            robot_manager = RobotManagerIGE(
                robot_config=self.robot_config,
                controller_config=self.controller_config,
                env_config=self.env_config,
                device=self.device,
                robot_id=robot_idx  # Add robot ID for tracking
            )
            self.robot_managers.append(robot_manager)
            
    def populate_env(self):
        # ... existing code ...
        
        for env_idx in range(self.num_envs):
            for robot_idx, robot_manager in enumerate(self.robot_managers):
                robot_manager.add_robot_to_env(env_idx, robot_idx)
                
    def step(self, actions):
        """
        Handle multi-agent actions
        actions: tensor of shape (num_envs, num_robots_per_env, action_dim)
        """
        for robot_idx, robot_manager in enumerate(self.robot_managers):
            # Extract actions for this robot across all environments
            robot_actions = actions[:, robot_idx, :]
            robot_manager.step(robot_actions)
            
    def get_observations(self):
        """
        Collect observations from all robots
        Returns: dict with observations for each robot
        """
        observations = {}
        for robot_idx, robot_manager in enumerate(self.robot_managers):
            observations[f"robot_{robot_idx}"] = robot_manager.get_observations()
        return observations 