from aerial_gym.task.navigation_task_gate.navigation_task_gate import NavigationTaskGate
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)

from aerial_gym.utils.math import quat_rotate_inverse, get_euler_xyz_tensor
import torch


class DCE_RL_Navigation_Task_Gate(NavigationTaskGate):
    def __init__(self, task_config, **kwargs):
        task_config.action_space_dim = 4  # 4D action space [x_vel, y_vel, z_vel, yaw_rate] for direct velocity control
        task_config.curriculum.min_level = 3  # Gate task starts from level 3 (matches gate environment obstacles)
        task_config.curriculum.max_level = 23  # Gate task goes up to level 23
        
        # Handle headless setting from Sample Factory command line parameters
        # Check if Sample Factory passed headless setting via environment variable
        import os
        sf_headless = os.environ.get('SF_HEADLESS', None)
        if sf_headless is not None:
            task_config.headless = sf_headless.lower() == 'true'
            logger.info(f"DCE Gate Navigation Task - Using SF_HEADLESS environment variable: {task_config.headless}")
        elif not hasattr(task_config, 'headless') or task_config.headless is None:
            task_config.headless = False  # Default to visualization enabled for gate navigation
            logger.info(f"DCE Gate Navigation Task - Using default headless=False for visualization")
        else:
            logger.info(f"DCE Gate Navigation Task - Using pre-configured headless: {task_config.headless}")
        
        logger.info(f"DCE Gate Navigation Task - Final headless mode: {task_config.headless}")
        
        # Check for Sample Factory env_agents parameter to force specific environment count  
        # This handles rollout worker subprocesses that don't go through registration
        env_agents_override = None
        try:
            # Try to access the global Sample Factory config if available
            import os
            if 'SF_ENV_AGENTS' in os.environ:
                env_agents_override = int(os.environ['SF_ENV_AGENTS'])
                logger.info(f"Found SF_ENV_AGENTS environment variable: {env_agents_override}")
        except:
            pass
        
        # Force specific environment count if env_agents is specified
        if env_agents_override is not None and env_agents_override > 0:
            logger.info(f"Detected env_agents={env_agents_override} from environment - setting environment count.")
            task_config.num_envs = env_agents_override
        else:
            logger.info(f"Using {task_config.num_envs} environments as configured.")
            
        super().__init__(task_config=task_config, **kwargs)

    # ===== ENHANCED: Complete drone state observations including absolute position =====
    # 150D total: 3D drone position + 6D static camera pose + 3D full orientation + 9D state + 64D drone VAE + 64D static camera VAE

    def process_obs_for_task(self):
        # MODIFIED: Include drone absolute position and full orientation sensing
        # This provides the agent with complete spatial awareness of its state and static camera relative position
        
        # ===== DRONE ABSOLUTE POSITION OBSERVATIONS (3D) =====
        # [0:3] = Drone absolute position in world coordinates (x, y, z)
        self.task_obs["observations"][:, 0:3] = self.obs_dict["robot_position"]
        
        # ===== STATIC CAMERA POSE OBSERVATIONS (6D) =====
        # Get static camera pose information relative to drone
        static_camera_pos, static_camera_orientation = self._get_static_camera_pose_relative_to_drone()
        
        # [3:6] = Static camera position relative to drone (x, y, z in drone's reference frame)
        self.task_obs["observations"][:, 3:6] = static_camera_pos
        
        # [6:9] = Static camera orientation relative to drone (roll, pitch, yaw in drone's reference frame)
        self.task_obs["observations"][:, 6:9] = static_camera_orientation
        
        # ===== DRONE FULL ORIENTATION OBSERVATIONS (3D) =====
        # [9:12] = Full drone orientation including yaw (roll, pitch, yaw)
        euler_angles = ssa(get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"]))
        self.task_obs["observations"][:, 9:12] = euler_angles  # MODIFIED: Include full yaw instead of setting to 0.0
        
        # ===== DRONE STATE OBSERVATIONS (10D) =====
        # [12:15] = Robot body linear velocity
        self.task_obs["observations"][:, 12:15] = self.obs_dict["robot_body_linvel"]
        
        # [15:18] = Robot body angular velocity  
        self.task_obs["observations"][:, 15:18] = self.obs_dict["robot_body_angvel"]
        
        # [18:22] = Robot actions (x_vel, y_vel, z_vel, yaw_rate)
        self.task_obs["observations"][:, 18:22] = self.obs_dict["robot_actions"]
        
        # ===== VISUAL OBSERVATIONS (128D) =====
        # [22:86] = Drone camera VAE latents (64D)
        self.task_obs["observations"][:, 22:86] = self.image_latents
        
        # [86:150] = Static camera VAE latents (64D)
        self.task_obs["observations"][:, 86:150] = self.static_image_latents

        # DEBUG: Periodic spawn/position logging to track drone movement
        if not hasattr(self, '_position_debug_counter'):
            self._position_debug_counter = 0
        
        self._position_debug_counter += 1
        
        # DEBUG: Detect fresh spawns (very low position and velocity)
        drone_pos = self.obs_dict["robot_position"]
        drone_linvel = self.obs_dict["robot_body_linvel"] 
        drone_angvel = self.obs_dict["robot_body_angvel"]
        
        # Check for spawn conditions: near origin + very low velocities
        position_norm = torch.norm(drone_pos, dim=1)
        linvel_norm = torch.norm(drone_linvel, dim=1)
        angvel_norm = torch.norm(drone_angvel, dim=1)
        
        # Detect likely spawns (position < 1m from origin, velocities < 0.1 m/s)
        likely_spawns = (position_norm < 1.0) & (linvel_norm < 0.1) & (angvel_norm < 0.1)
        
        if torch.any(likely_spawns) and not hasattr(self, '_spawn_detected'):
            spawn_envs = torch.where(likely_spawns)[0]
            logger.warning("="*80)
            logger.warning(f"🎯 SPAWN DETECTION: Found {len(spawn_envs)} likely spawned environments")
            for env_id in spawn_envs[:3]:  # Log first 3 for brevity
                pos = drone_pos[env_id]
                vel = drone_linvel[env_id]
                euler = ssa(get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"]))[env_id]
                static_rel = static_camera_pos[env_id]
                logger.warning(f"  🚁 Env {env_id}: Pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], Vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")
                logger.warning(f"            Orient=[{euler[0]*57.3:.1f}°, {euler[1]*57.3:.1f}°, {euler[2]*57.3:.1f}°]")
                logger.warning(f"            Static Rel=[{static_rel[0]:.3f}, {static_rel[1]:.3f}, {static_rel[2]:.3f}]")
            logger.warning("="*80)
            self._spawn_detected = True  # Only log spawns once per session

    def _get_static_camera_pose_relative_to_drone(self):
        """
        Calculate static camera position and orientation relative to the drone's reference frame.
        
        Returns:
            static_camera_pos: [num_envs, 3] - Camera position in drone's reference frame
            static_camera_orientation: [num_envs, 3] - Camera orientation (roll, pitch, yaw) relative to drone
        """
        # Static camera base position (world coordinates)
        # Camera is at (0.0, -3.0, 1.5) in world frame, but orientation varies by curriculum
        static_camera_world_pos = torch.tensor([0.0, -3.0, 1.5], device=self.device).expand(self.num_envs, -1)
        
        # Get drone position and orientation
        drone_pos = self.obs_dict["robot_position"]  # [num_envs, 3]
        drone_orientation = self.obs_dict["robot_vehicle_orientation"]  # [num_envs, 4] quaternion
        
        # DEBUG: Log coordinate transformation for first environment
        if not hasattr(self, '_coord_debug_logged'):
            self._coord_debug_logged = True
            
            # Convert quaternion to euler for readable display
            from aerial_gym.utils.math import get_euler_xyz_tensor, ssa
            drone_euler = ssa(get_euler_xyz_tensor(drone_orientation))
            
            logger.warning("="*80)
            logger.warning("🌍 COORDINATE TRANSFORMATION DEBUG:")
            logger.warning(f"  📍 Static Camera World Position (FIXED): {static_camera_world_pos[0].cpu().numpy()}")
            logger.warning(f"  🤖 Drone World Position (DYNAMIC): {drone_pos[0].cpu().numpy()}")
            logger.warning(f"  🧭 Drone World Orientation (Quaternion): {drone_orientation[0].cpu().numpy()}")
            logger.warning(f"  🧭 Drone World Orientation (Euler deg): [{drone_euler[0, 0].item()*57.3:.1f}°, {drone_euler[0, 1].item()*57.3:.1f}°, {drone_euler[0, 2].item()*57.3:.1f}°]")
            
            # Calculate intermediate steps for debugging
            camera_pos_world_relative = static_camera_world_pos - drone_pos
            logger.warning(f"  ↔️  World Vector (camera - drone): {camera_pos_world_relative[0].cpu().numpy()}")
            
            # Show the transformation
            static_camera_pos = quat_rotate_inverse(drone_orientation, camera_pos_world_relative)
            logger.warning(f"  🔄 After Quaternion Transform: {static_camera_pos[0].cpu().numpy()}")
            logger.warning(f"  📏 Distance from drone to camera: {torch.norm(static_camera_pos[0]).item():.3f}m")
            logger.warning("="*80)
        
        # Calculate camera position relative to drone
        camera_pos_world_relative = static_camera_world_pos - drone_pos
        
        # Transform camera position to drone's reference frame
        static_camera_pos = quat_rotate_inverse(drone_orientation, camera_pos_world_relative)
        
        # Calculate camera orientation relative to drone
        static_camera_orientation = self._calculate_camera_orientation_relative_to_drone()
        
        return static_camera_pos, static_camera_orientation
    
    def _calculate_camera_orientation_relative_to_drone(self):
        """
        Calculate static camera orientation relative to drone's reference frame.
        
        The static camera orientation depends on the curriculum level and current camera angles.
        Returns the camera's orientation as Euler angles relative to the drone.
        
        Returns:
            camera_orientation: [num_envs, 3] - Camera orientation (roll, pitch, yaw) relative to drone
        """
        # Get current camera angles for each environment (stored by StaticCameraManager)
        if hasattr(self, 'static_camera_manager') and hasattr(self.static_camera_manager, 'current_camera_angles'):
            camera_angles = self.static_camera_manager.current_camera_angles
        else:
            # Fallback: assume 0 degree angle for all environments
            camera_angles = [0.0] * self.num_envs
        
        # Ensure we have angles for all environments
        while len(camera_angles) < self.num_envs:
            camera_angles.append(0.0)
        
        # Convert camera angles to tensor
        camera_yaw_angles = torch.tensor(camera_angles[:self.num_envs], device=self.device)
        
        # Get drone's current orientation (Euler angles)
        drone_euler = ssa(get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"]))
        
        # Camera orientation in world frame (looking at gate with curriculum-dependent angle)
        # Camera base orientation: looking forward (0° roll, 0° pitch, variable yaw based on curriculum)
        camera_roll_world = torch.zeros_like(camera_yaw_angles)
        camera_pitch_world = torch.zeros_like(camera_yaw_angles)
        camera_yaw_world = camera_yaw_angles * (3.14159 / 180.0)  # Convert degrees to radians
        
        # Calculate relative orientation (camera orientation - drone orientation)
        camera_roll_relative = camera_roll_world - drone_euler[:, 0]
        camera_pitch_relative = camera_pitch_world - drone_euler[:, 1] 
        camera_yaw_relative = camera_yaw_world - drone_euler[:, 2]
        
        # Normalize angles to [-π, π]
        camera_roll_relative = ssa(camera_roll_relative)
        camera_pitch_relative = ssa(camera_pitch_relative)
        camera_yaw_relative = ssa(camera_yaw_relative)
        
        # Stack into [num_envs, 3] tensor
        static_camera_orientation = torch.stack([
            camera_roll_relative, 
            camera_pitch_relative, 
            camera_yaw_relative
        ], dim=1)
        
        return static_camera_orientation


@torch.jit.script
def ssa(a: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle"""
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi