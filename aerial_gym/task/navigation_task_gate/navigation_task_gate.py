from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

from aerial_gym.utils.vae.vae_image_encoder import VAEImageEncoder

import gymnasium as gym
from gym.spaces import Dict, Box

# Isaac Gym imports for static camera management
from isaacgym import gymapi, gymtorch

logger = CustomLogger("navigation_task_gate")


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class NavigationTaskGate(BaseTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        # overwrite the params if user has provided them
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp
        super().__init__(task_config)
        self.device = self.task_config.device
        
        # set the each of the elements of reward parameter to a torch tensor
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
        logger.info("Building environment for gate navigation task.")
        logger.info(
            "Sim Name: {}, Env Name: {}, Robot Name: {}, Controller Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        self.target_min_ratio = torch.tensor(
            self.task_config.target_min_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs, -1)
        self.target_max_ratio = torch.tensor(
            self.task_config.target_max_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs, -1)

        self.success_aggregate = 0
        self.crashes_aggregate = 0
        self.timeouts_aggregate = 0

        self.pos_error_vehicle_frame_prev = torch.zeros_like(self.target_position)
        self.pos_error_vehicle_frame = torch.zeros_like(self.target_position)

        # Gate-specific tracking
        self.gate_position = torch.zeros((self.sim_env.num_envs, 3), device=self.device)
        self.gate_passed = torch.zeros(self.sim_env.num_envs, device=self.device, dtype=torch.bool)
        self.gate_approach_distance = torch.zeros(self.sim_env.num_envs, device=self.device)

        # Initialize single shared VAE model for both drone and static cameras
        # This optimization reduces GPU memory usage by ~50% compared to loading two separate models
        if self.task_config.vae_config.use_vae:
            self.shared_vae_model = VAEImageEncoder(config=self.task_config.vae_config, device=self.device)
            self.image_latents = torch.zeros(
                (self.sim_env.num_envs, self.task_config.vae_config.latent_dims),
                device=self.device,
                requires_grad=False,
            )
            # Reuse the same VAE model for static camera processing
            self.static_camera_latents = torch.zeros(
                (self.sim_env.num_envs, self.task_config.vae_config.latent_dims),  # Same latent dims
                device=self.device,
                requires_grad=False,
            )
        else:
            self.shared_vae_model = lambda x: x
            self.image_latents = torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            )
            self.static_camera_latents = torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            )

        # Static camera management using Isaac Gym native API
        self.static_camera_manager = StaticCameraManager(self.sim_env, self.task_config)

        # Get the dictionary once from the environment and use it to get the observations later.
        self.obs_dict = self.sim_env.get_obs()
        if "curriculum_level" not in self.obs_dict.keys():
            self.curriculum_level = self.task_config.curriculum.min_level
            self.obs_dict["curriculum_level"] = self.curriculum_level
        else:
            self.curriculum_level = self.obs_dict["curriculum_level"]
        
        # Calculate total obstacles: fixed assets (1 gate + 6 walls = 7) + curriculum objects (3)
        # Asset manager expects count of ALL assets with keep_in_env=True (total=10)
        FIXED_ASSETS_COUNT = 7  # 1 gate + 6 boundary walls (left, right, front, back, bottom, top)
        total_obstacles_in_env = FIXED_ASSETS_COUNT + self.curriculum_level  # 7 + 3 = 10
        self.obs_dict["num_obstacles_in_env"] = total_obstacles_in_env
        self.curriculum_progress_fraction = (
            self.curriculum_level - self.task_config.curriculum.min_level
        ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        # Enhanced observation space for gate navigation with static camera
        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.task_config.observation_space_dim,),  # 145D: 17D basic + 64D drone VAE + 64D static VAE
                    dtype=np.float32,
                ),
                "image_obs": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(1, 135, 240),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)  # 4D action space
        self.action_transformation_function = self.task_config.action_transformation_function

        self.num_envs = self.sim_env.num_envs

        # Enhanced task observations for gate navigation
        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.sim_env.num_envs,
                    self.task_config.privileged_observation_space_dim,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

        self.num_task_steps = 0

    def close(self):
        try:
            if hasattr(self.sim_env, 'delete_env'):
                self.sim_env.delete_env()
            elif hasattr(self.sim_env, 'close'):
                self.sim_env.close()
            else:
                print("[DEBUG] No cleanup method found for sim_env")
        except Exception as e:
            print(f"[DEBUG] Error during close: {e}")

    def reset(self):
        self.reset_idx(torch.arange(self.sim_env.num_envs))
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        # Set target positions (goals remain on front side of gate)
        target_ratio = torch_rand_float_tensor(self.target_min_ratio, self.target_max_ratio)
        self.target_position[env_ids] = torch_interpolate_ratio(
            min=self.obs_dict["env_bounds_min"][env_ids],
            max=self.obs_dict["env_bounds_max"][env_ids],
            ratio=target_ratio[env_ids],
        )
        
        # Reset gate-specific tracking
        self.gate_passed[env_ids] = False
        self.gate_approach_distance[env_ids] = 0.0
        
        # Update static camera position based on curriculum level
        self.static_camera_manager.update_camera_positions(self.curriculum_level, env_ids)
        
        self.infos = {}
        return

    def render(self):
        return self.sim_env.render()

    def step(self, actions):
        # UPDATED: Transform 4D actions to 4D velocity commands for X500 robot with Z-axis control
        # Input: [x_vel_cmd, y_vel_cmd, z_vel_cmd, yaw_rate_cmd] ∈ [-1, 1]^4
        # Output: [x_vel, y_vel, z_vel, yaw_rate] in real units
        
        # Apply action transformation function from task config (4D -> 4D)
        transformed_action = self.action_transformation_function(actions)
        logger.debug(f"raw_action: {actions[0]}, transformed action: {transformed_action[0]}")
        
        # Pass 4D velocity commands directly to simulation environment
        self.sim_env.step(actions=transformed_action)

        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        self.rewards[:], self.terminations[:], camera_gate_alignment = self.compute_rewards_and_crashes(self.obs_dict)

        # logger.info(f"Curricluum Level: {self.curriculum_level}")

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        # successes are are the sum of the environments which are to be truncated and have reached the target within a distance threshold
        successes = self.truncations * (
            torch.norm(self.target_position - self.obs_dict["robot_position"], dim=1) < 1.0
        )
        successes = torch.where(self.terminations > 0, torch.zeros_like(successes), successes)
        timeouts = torch.where(
            self.truncations > 0, torch.logical_not(successes), torch.zeros_like(successes)
        )
        timeouts = torch.where(
            self.terminations > 0, torch.zeros_like(timeouts), timeouts
        )  # timeouts are not counted if there is a crash

        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = self.terminations
        
        # Add gate navigation specific info to wandb tracking
        # Calculate gate navigation metrics from current state
        robot_position = self.obs_dict["robot_position"]
        gate_distance = torch.norm(robot_position - self.gate_position, dim=1)
        
        # Check if robot has passed gate (crossed Y = 0 plane with proper alignment)
        gate_passed_current = (
            (robot_position[:, 1] > self.gate_position[:, 1]) &  # In front of gate
            (torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < 1.5) &  # Within gate width
            (robot_position[:, 2] > 0.2) & (robot_position[:, 2] < 2.2)  # Within gate height
        )
        
        # Gate alignment: check if robot is roughly aligned with gate opening
        gate_alignment = torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < 1.5
        
        # Camera alignment angle in degrees (convert from dot product)
        alignment_angle_deg = torch.acos(torch.clamp(camera_gate_alignment, -1.0, 1.0)) * 180.0 / 3.14159
        
        # Camera alignment category based on angle
        alignment_category = torch.zeros_like(alignment_angle_deg)
        alignment_category[alignment_angle_deg <= 15] = 5  # Perfect
        alignment_category[(alignment_angle_deg > 15) & (alignment_angle_deg <= 30)] = 4  # Excellent
        alignment_category[(alignment_angle_deg > 30) & (alignment_angle_deg <= 60)] = 3  # Good
        alignment_category[(alignment_angle_deg > 60) & (alignment_angle_deg <= 90)] = 2  # Moderate
        alignment_category[(alignment_angle_deg > 90) & (alignment_angle_deg <= 135)] = 1  # Poor
        alignment_category[alignment_angle_deg > 135] = 0  # Severely misaligned
        
        self.infos["gate/passed"] = gate_passed_current.float()
        self.infos["gate/distance"] = gate_distance
        self.infos["gate/alignment"] = gate_alignment.float()
        self.infos["camera/facing_alignment"] = camera_gate_alignment
        self.infos["camera/alignment_angle_deg"] = alignment_angle_deg
        self.infos["camera/alignment_category"] = alignment_category
        
        # Add continuous curriculum tracking for wandb
        self.infos["curriculum/current_level"] = torch.tensor(self.curriculum_level, dtype=torch.float32)
        self.infos["curriculum/current_progress"] = torch.tensor(self.curriculum_progress_fraction, dtype=torch.float32)

        self.logging_sanity_check(self.infos)
        self.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )
        # rendering happens at the post-reward calculation step since the newer measurement is required to be
        # sent to the RL algorithm as an observation and it helps if the camera image is updated then
        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)
        self.num_task_steps += 1
        # do stuff with the image observations here
        self.process_image_observation()
        self.process_static_camera_observation()
        self.post_image_reward_addition()
        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()
        return return_tuple

    def process_image_observation(self):
        """Process drone camera observations."""
        image_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
        if self.task_config.vae_config.use_vae:
            self.image_latents[:] = self.shared_vae_model.encode(image_obs)

    def process_static_camera_observation(self):
        """Process static camera observations."""
        try:
            static_depth, static_seg = self.static_camera_manager.capture_images()
            if static_depth is not None and self.task_config.vae_config.use_vae:
                # Convert to tensor and process through VAE
                if isinstance(static_depth, np.ndarray):
                    static_depth_tensor = torch.from_numpy(static_depth).float().to(self.device)
                    if static_depth_tensor.dim() == 2:
                        static_depth_tensor = static_depth_tensor.unsqueeze(0)  # Add batch dimension
                    # Ensure all environments get the same static camera view
                    static_depth_expanded = static_depth_tensor.expand(self.sim_env.num_envs, -1, -1)
                    self.static_camera_latents[:] = self.shared_vae_model.encode(static_depth_expanded)
                else:
                    # If already tensor
                    if static_depth.dim() == 2:
                        static_depth = static_depth.unsqueeze(0)
                    static_depth_expanded = static_depth.expand(self.sim_env.num_envs, -1, -1)
                    self.static_camera_latents[:] = self.shared_vae_model.encode(static_depth_expanded)
        except Exception as e:
            logger.debug(f"Static camera processing error: {e}")

    def post_image_reward_addition(self):
        """Add image-based rewards from drone camera."""
        image_obs = 10.0 * self.obs_dict["depth_range_pixels"].squeeze(1)
        image_obs[image_obs < 0] = 10.0
        self.min_pixel_dist = torch.amin(image_obs, dim=(1, 2))
        self.rewards[self.terminations < 0] += -exponential_reward_function(
            4.0, 1.0, self.min_pixel_dist[self.terminations < 0]
        )

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        """Process observations for gate navigation task with enhanced observation space."""
        vec_to_tgt = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )
        dist_to_tgt = torch.norm(vec_to_tgt, dim=-1)
        
        # Basic state observations (17D)
        self.task_obs["observations"][:, 0:3] = vec_to_tgt / dist_to_tgt.unsqueeze(1)
        self.task_obs["observations"][:, 3] = dist_to_tgt / 5.0
        
        euler_angles = ssa(get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"]))
        self.task_obs["observations"][:, 4:6] = euler_angles[:, 0:2]
        self.task_obs["observations"][:, 6] = 0.0
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        # UPDATED: Handle 4D actions [x_vel, y_vel, z_vel, yaw_rate] instead of 3D
        self.task_obs["observations"][:, 13:17] = self.obs_dict["robot_actions"]  # All 4 actions now
        
        # Drone camera VAE latents (64D)
        if self.task_config.vae_config.use_vae:
            self.task_obs["observations"][:, 17:81] = self.image_latents
        
        # Static camera VAE latents (64D)
        if self.task_config.vae_config.use_vae:
            self.task_obs["observations"][:, 81:145] = self.static_camera_latents

        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations
        self.task_obs["image_obs"] = self.obs_dict["depth_range_pixels"]

    def compute_rewards_and_crashes(self, obs_dict):
        """Compute rewards with gate-specific components."""
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        
        self.pos_error_vehicle_frame_prev[:] = self.pos_error_vehicle_frame
        self.pos_error_vehicle_frame[:] = quat_rotate_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        
        rewards, crashes, camera_gate_alignment = compute_gate_reward(
            self.pos_error_vehicle_frame,
            self.pos_error_vehicle_frame_prev,
            obs_dict["crashes"],
            obs_dict["robot_actions"],
            obs_dict["robot_prev_actions"],
            robot_position,
            robot_vehicle_orientation,
            self.gate_position,
            self.gate_passed,
            self.curriculum_progress_fraction,
            self.task_config.reward_parameters,
        )
        
        # Store camera alignment for debugging
        self.camera_alignment_debug = camera_gate_alignment
        
        return rewards, crashes, camera_gate_alignment

    def check_and_update_curriculum_level(self, successes, crashes, timeouts):
        """Update curriculum level and static camera positioning."""
        self.success_aggregate += torch.sum(successes)
        self.crashes_aggregate += torch.sum(crashes)
        self.timeouts_aggregate += torch.sum(timeouts)

        instances = self.success_aggregate + self.crashes_aggregate + self.timeouts_aggregate

        if instances >= self.task_config.curriculum.check_after_log_instances:
            success_rate = self.success_aggregate / instances
            crash_rate = self.crashes_aggregate / instances
            timeout_rate = self.timeouts_aggregate / instances

            if success_rate > self.task_config.curriculum.success_rate_for_increase:
                self.curriculum_level += self.task_config.curriculum.increase_step
            elif success_rate < self.task_config.curriculum.success_rate_for_decrease:
                self.curriculum_level -= self.task_config.curriculum.decrease_step

            # clamp curriculum_level
            self.curriculum_level = min(
                max(self.curriculum_level, self.task_config.curriculum.min_level),
                self.task_config.curriculum.max_level,
            )
            self.obs_dict["curriculum_level"] = self.curriculum_level
            
            # Calculate total obstacles: fixed assets (1 gate + 6 walls = 7) + curriculum objects (3)
            # Asset manager expects count of ALL assets with keep_in_env=True (total=10)
            FIXED_ASSETS_COUNT = 7  # 1 gate + 6 boundary walls (left, right, front, back, bottom, top)
            total_obstacles_in_env = FIXED_ASSETS_COUNT + self.curriculum_level  # 7 + 3 = 10
            self.obs_dict["num_obstacles_in_env"] = total_obstacles_in_env
            self.curriculum_progress_fraction = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

            logger.warning(
                f"Gate Navigation Curriculum Level: {self.curriculum_level}, Progress: {self.curriculum_progress_fraction}"
            )
            logger.warning(
                f"\nSuccess Rate: {success_rate}\nCrash Rate: {crash_rate}\nTimeout Rate: {timeout_rate}"
            )
            
            # Add curriculum metrics to infos for wandb logging
            self.infos["curriculum/level"] = torch.tensor(self.curriculum_level, dtype=torch.float32)
            self.infos["curriculum/progress"] = torch.tensor(self.curriculum_progress_fraction, dtype=torch.float32)
            self.infos["curriculum/success_rate"] = torch.tensor(success_rate, dtype=torch.float32)
            self.infos["curriculum/crash_rate"] = torch.tensor(crash_rate, dtype=torch.float32)
            self.infos["curriculum/timeout_rate"] = torch.tensor(timeout_rate, dtype=torch.float32)
            
            self.success_aggregate = 0
            self.crashes_aggregate = 0
            self.timeouts_aggregate = 0

    def logging_sanity_check(self, infos):
        """Logging sanity check for gate navigation."""
        successes = infos["successes"]
        crashes = infos["crashes"]
        timeouts = infos["timeouts"]
        
        if torch.sum(torch.logical_and(successes, crashes)) > 0:
            logger.critical("Success and crash are occuring at the same time")
        if torch.sum(torch.logical_and(successes, timeouts)) > 0:
            logger.critical("Success and timeout are occuring at the same time")
        if torch.sum(torch.logical_and(crashes, timeouts)) > 0:
            logger.critical("Crash and timeout are occuring at the same time")


class StaticCameraManager:
    """Manages static camera for gate navigation using Isaac Gym native API."""
    
    def __init__(self, env_manager, task_config):
        self.env_manager = env_manager
        self.task_config = task_config
        self.gym = env_manager.IGE_env.gym
        self.sim = env_manager.IGE_env.sim
        self.env_handles = env_manager.IGE_env.env_handles
        self.camera_handles = []
        self.camera_setup_success = False
        self.use_synthetic_camera = False  # Initialize synthetic camera flag
        
        # Gate position (center of environment)
        self.gate_position = [0.0, 0.0, 0.0]
        self.env_bounds = [[-4.0, -4.0, 0.0], [4.0, 4.0, 4.0]]  # Updated for gate_env bounds
        
        self._setup_static_camera()
    
    def _setup_static_camera(self):
        """Setup static camera using Isaac Gym native camera API with D455 specifications."""
        logger.info("Setting up static camera for gate navigation...")
        
        # Check if simulation is running in headless mode
        if self.task_config.headless:
            logger.info("Running in headless mode - static camera will use synthetic data for training")
            self.camera_setup_success = False
            self.use_synthetic_camera = True
            return
        
        try:
            # Camera properties (D455 depth camera specifications - match working example)
            camera_props = gymapi.CameraProperties()
            camera_props.width = 480  # D455 depth resolution width
            camera_props.height = 270  # D455 depth resolution height
            camera_props.horizontal_fov = 87.0  # D455 FOV
            camera_props.near_plane = 0.4  # D455 minimum depth distance
            camera_props.far_plane = 20.0  # D455 maximum range
            camera_props.enable_tensors = True  # Enable GPU tensor access
            
            logger.info(f"Static camera properties (D455 specs): {camera_props.width}x{camera_props.height}, FOV: {camera_props.horizontal_fov}°")
            logger.info(f"Static camera depth range: {camera_props.near_plane}m - {camera_props.far_plane}m")
        
            # Create camera sensor in each environment
            self.camera_handles = []
            for i, env_handle in enumerate(self.env_handles):
                cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                if cam_handle >= 0:  # Valid camera handle
                    self.camera_handles.append(cam_handle)
                    logger.info(f"Created static camera sensor {i} in environment {i}")
                else:
                    logger.warning(f"Failed to create camera for environment {i}, handle: {cam_handle}")
                    self.camera_setup_success = False
                    self.use_synthetic_camera = True
                    return
            
            # FIXED POSITIONING: Position camera to face the gate directly (match working example)
            # Gate is positioned at ground level (Z=0), camera at 1.5m height looking at gate center
            camera_pos = gymapi.Vec3(0.0, -3.0, 1.5)  # 3m behind gate, at gate center height
            camera_target = gymapi.Vec3(0.0, 0.0, 1.5)  # Look directly at gate center
            
            # Set camera transform for each environment using fixed positioning
            for i, (env_handle, cam_handle) in enumerate(zip(self.env_handles, self.camera_handles)):
                # Use Isaac Gym's camera look_at functionality (match working example)
                self.gym.set_camera_location(cam_handle, env_handle, camera_pos, camera_target)
                logger.info(f"Set static camera {i} to look from ({camera_pos.x}, {camera_pos.y}, {camera_pos.z}) toward ({camera_target.x}, {camera_target.y}, {camera_target.z})")
            
            logger.info("✓ Static camera setup complete with fixed positioning")
            self.camera_setup_success = True
            self.use_synthetic_camera = False
            
        except Exception as e:
            logger.warning(f"Static camera setup failed, falling back to synthetic data: {e}")
            self.camera_setup_success = False
            self.use_synthetic_camera = True
    
    def update_camera_positions(self, curriculum_level, env_ids):
        """Update static camera positions based on curriculum level."""
        if hasattr(self, 'use_synthetic_camera') and self.use_synthetic_camera:
            # In synthetic mode, just log the curriculum level for reference
            logger.debug(f"Synthetic camera mode - curriculum level {curriculum_level}")
            return
            
        if not self.camera_setup_success:
            return
        
        # FIXED POSITIONING: Camera position is now fixed during setup, no dynamic repositioning needed
        logger.debug(f"Static camera uses fixed positioning - curriculum level {curriculum_level} noted but position unchanged")
    
    def capture_images(self):
        """Capture depth and segmentation images from static camera."""
        if hasattr(self, 'use_synthetic_camera') and self.use_synthetic_camera:
            # Generate synthetic camera data for headless training
            return self._generate_synthetic_camera_data()
        
        if not self.camera_setup_success or len(self.camera_handles) == 0:
            return self._generate_synthetic_camera_data()
        
        try:
            # Step graphics and render all cameras
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            
            # Get images from first camera
            env_handle = self.env_handles[0]
            cam_handle = self.camera_handles[0]
            
            # Get depth image
            depth_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH
            )
            depth_img = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
            
            # Get segmentation image
            seg_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, env_handle, cam_handle, gymapi.IMAGE_SEGMENTATION
            )
            seg_img = gymtorch.wrap_tensor(seg_tensor).cpu().numpy()
            
            # End access to image tensors
            self.gym.end_access_image_tensors(self.sim)
            
            # Process depth for VAE (match working example processing)
            if depth_img is not None:
                # Convert to DCE format for consistency with robot camera processing
                # Static camera gives raw depth values, need to normalize to [0,1] for DCE processing
                depth_normalized = depth_img.copy()
                depth_normalized[depth_normalized == -np.inf] = 20.0  # Use far_plane value
                depth_normalized = np.abs(depth_normalized)  # Handle negative depths
                depth_normalized = np.clip(depth_normalized, 0.4, 20.0)  # Clip to camera range
                # Normalize to [0,1] range like DCE navigation expects
                depth_normalized = (depth_normalized - 0.4) / (20.0 - 0.4)
                depth_img = depth_normalized.astype(np.float32)
            
            return depth_img, seg_img
                
        except Exception as e:
            logger.debug(f"Static camera capture error, falling back to synthetic: {e}")
            return self._generate_synthetic_camera_data()
    
    def _generate_synthetic_camera_data(self):
        """Generate synthetic camera data for headless training."""
        try:
            # Create synthetic depth image (480x270) with reasonable gate-like features
            height, width = 270, 480
            depth_img = np.full((height, width), 0.5, dtype=np.float32)  # Mid-range depth
            
            # Add gate-like features to the synthetic depth
            # Create a rectangular opening (gate) in the center
            gate_x_start = width // 2 - 60  # Gate width ~120 pixels
            gate_x_end = width // 2 + 60
            gate_y_start = height // 2 - 40  # Gate height ~80 pixels
            gate_y_end = height // 2 + 40
            
            # Gate opening (closer depth)
            depth_img[gate_y_start:gate_y_end, gate_x_start:gate_x_end] = 0.8
            
            # Gate frame (farther depth)
            frame_thickness = 10
            # Top and bottom frame
            depth_img[gate_y_start-frame_thickness:gate_y_start, gate_x_start-frame_thickness:gate_x_end+frame_thickness] = 0.2
            depth_img[gate_y_end:gate_y_end+frame_thickness, gate_x_start-frame_thickness:gate_x_end+frame_thickness] = 0.2
            # Left and right frame
            depth_img[gate_y_start:gate_y_end, gate_x_start-frame_thickness:gate_x_start] = 0.2
            depth_img[gate_y_start:gate_y_end, gate_x_end:gate_x_end+frame_thickness] = 0.2
            
            # Add some noise for realism
            noise = np.random.normal(0, 0.02, (height, width)).astype(np.float32)
            depth_img = np.clip(depth_img + noise, 0.0, 1.0)
            
            # Create synthetic segmentation image
            seg_img = np.zeros((height, width), dtype=np.uint8)
            seg_img[gate_y_start:gate_y_end, gate_x_start:gate_x_end] = 1  # Gate opening
            
            return depth_img, seg_img
            
        except Exception as e:
            logger.debug(f"Synthetic camera data generation error: {e}")
            # Return zero arrays as fallback
            return np.zeros((270, 480), dtype=np.float32), np.zeros((270, 480), dtype=np.uint8)


@torch.jit.script
def exponential_reward_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward function"""
    return magnitude * torch.exp(-(value * value) * exponent)


@torch.jit.script
def exponential_penalty_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential penalty function"""
    return magnitude * (torch.exp(-(value * value) * exponent) - 1.0)


@torch.jit.script
def ssa(a: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle"""
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi


@torch.jit.script
def compute_gate_reward(
    pos_error,
    prev_pos_error,
    crashes,
    action,
    prev_action,
    robot_position,
    robot_vehicle_orientation,
    gate_position,
    gate_passed,
    curriculum_progress_fraction,
    parameter_dict,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    
    # Base reward computation - REDUCED multiplication factor to prevent over-rewarding
    MULTIPLICATION_FACTOR_REWARD = 1.0 + (0.5) * curriculum_progress_fraction  # Reduced from 2.0 to 0.5
    dist = torch.norm(pos_error, dim=1)
    prev_dist_to_goal = torch.norm(prev_pos_error, dim=1)
    
    pos_reward = exponential_reward_function(
        parameter_dict["pos_reward_magnitude"],
        parameter_dict["pos_reward_exponent"],
        dist,
    )
    very_close_to_goal_reward = exponential_reward_function(
        parameter_dict["very_close_to_goal_reward_magnitude"],
        parameter_dict["very_close_to_goal_reward_exponent"],
        dist,
    )

    getting_closer = prev_dist_to_goal - dist
    getting_closer_reward = torch.where(
        getting_closer > 0,
        parameter_dict["getting_closer_reward_multiplier"] * getting_closer,
        2.0 * parameter_dict["getting_closer_reward_multiplier"] * getting_closer,
    )

    # FIXED: Remove the problematic free distance reward that was giving ~1 point per step
    # distance_from_goal_reward = (20.0 - dist) / 20.0  # This was causing rapid learning!
    distance_from_goal_reward = torch.zeros_like(dist)  # Replace with zero reward
    
    # Action penalties
    action_diff = action - prev_action
    x_diff_penalty = exponential_penalty_function(
        parameter_dict["x_action_diff_penalty_magnitude"],
        parameter_dict["x_action_diff_penalty_exponent"],
        action_diff[:, 0],
    )
    z_diff_penalty = exponential_penalty_function(
        parameter_dict["z_action_diff_penalty_magnitude"],
        parameter_dict["z_action_diff_penalty_exponent"],
        action_diff[:, 2] if action_diff.shape[1] > 2 else torch.zeros_like(action_diff[:, 0]),
    )
    yawrate_diff_penalty = exponential_penalty_function(
        parameter_dict["yawrate_action_diff_penalty_magnitude"],
        parameter_dict["yawrate_action_diff_penalty_exponent"],
        action_diff[:, 2] if action_diff.shape[1] == 3 else action_diff[:, 3] if action_diff.shape[1] > 3 else torch.zeros_like(action_diff[:, 0]),
    )
    
    action_diff_penalty = x_diff_penalty + z_diff_penalty + yawrate_diff_penalty
    
    # Absolute action penalties
    x_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["x_absolute_action_penalty_magnitude"],
        parameter_dict["x_absolute_action_penalty_exponent"],
        action[:, 0],
    )
    z_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["z_absolute_action_penalty_magnitude"],
        parameter_dict["z_absolute_action_penalty_exponent"],
        action[:, 2] if action.shape[1] > 2 else torch.zeros_like(action[:, 0]),
    )
    yawrate_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["yawrate_absolute_action_penalty_magnitude"],
        parameter_dict["yawrate_absolute_action_penalty_exponent"],
        action[:, 2] if action.shape[1] == 3 else action[:, 3] if action.shape[1] > 3 else torch.zeros_like(action[:, 0]),
    )
    
    absolute_action_penalty = x_absolute_penalty + z_absolute_penalty + yawrate_absolute_penalty
    total_action_penalty = action_diff_penalty + absolute_action_penalty

    # Gate-specific rewards
    gate_distance = torch.norm(robot_position - gate_position, dim=1)
    
    # Reward for approaching gate
    gate_approach_reward = exponential_reward_function(
        parameter_dict["gate_approach_reward_magnitude"],
        0.5,
        gate_distance,
    )
    
    # Enhanced Camera Facing Reward System - Proportional to alignment angle
    # Calculate vector from drone to gate
    drone_to_gate = gate_position - robot_position
    drone_to_gate_normalized = drone_to_gate / (torch.norm(drone_to_gate, dim=1, keepdim=True) + 1e-8)
    
    # Get drone's forward direction (where camera points)
    # Camera faces forward in drone's body frame (+X direction after orientation)
    # Convert quaternion to rotation matrix and extract forward direction
    qw, qx, qy, qz = robot_vehicle_orientation[:, 3], robot_vehicle_orientation[:, 0], robot_vehicle_orientation[:, 1], robot_vehicle_orientation[:, 2]
    
    # Forward direction in world frame (drone's +X axis)
    forward_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    forward_y = 2.0 * (qx * qy + qw * qz)
    forward_z = 2.0 * (qx * qz - qw * qy)
    drone_forward = torch.stack([forward_x, forward_y, forward_z], dim=1)
    drone_forward_normalized = drone_forward / (torch.norm(drone_forward, dim=1, keepdim=True) + 1e-8)
    
    # Calculate alignment between camera direction and gate direction
    camera_gate_alignment = torch.sum(drone_forward_normalized * drone_to_gate_normalized, dim=1)
    camera_gate_alignment = torch.clamp(camera_gate_alignment, -1.0, 1.0)  # Clamp to [-1, 1]
    
    # Enhanced Proportional Camera Facing Reward System
    # alignment = 1.0  → facing directly toward gate (0° angle)
    # alignment = 0.966 → 15° angle  
    # alignment = 0.866 → 30° angle
    # alignment = 0.707 → 45° angle  
    # alignment = 0.5 → 60° angle
    # alignment = 0.0  → perpendicular (90° angle)
    # alignment = -0.707 → 135° angle
    # alignment = -1.0 → facing directly away (180° angle)
    
    camera_facing_reward = torch.zeros_like(camera_gate_alignment)
    
    # PERFECT ALIGNMENT: 0-15° (alignment > 0.966) - Maximum reward
    perfect_mask = camera_gate_alignment > 0.966  # cos(15°) ≈ 0.966
    camera_facing_reward[perfect_mask] = parameter_dict["camera_facing_reward_magnitude"]  # Full 5.0 reward
    
    # EXCELLENT ALIGNMENT: 15-30° (0.866 < alignment ≤ 0.966) - High reward
    excellent_mask = (camera_gate_alignment > 0.866) & (camera_gate_alignment <= 0.966)  # cos(30°) = 0.866
    camera_facing_reward[excellent_mask] = 0.9 * parameter_dict["camera_facing_reward_magnitude"] * camera_gate_alignment[excellent_mask]
    
    # GOOD ALIGNMENT: 30-60° (0.5 < alignment ≤ 0.866) - High reward  
    good_mask = (camera_gate_alignment > 0.5) & (camera_gate_alignment <= 0.866)  # cos(60°) = 0.5
    camera_facing_reward[good_mask] = 0.8 * parameter_dict["camera_facing_reward_magnitude"] * camera_gate_alignment[good_mask]
    
    # MODERATE ALIGNMENT: 60-90° (0 < alignment ≤ 0.5) - Moderate reward
    moderate_mask = (camera_gate_alignment > 0.0) & (camera_gate_alignment <= 0.5)
    camera_facing_reward[moderate_mask] = 0.4 * parameter_dict["camera_facing_reward_magnitude"] * camera_gate_alignment[moderate_mask]
    
    # POOR ALIGNMENT: 90-135° (-0.707 < alignment ≤ 0) - Small penalty
    poor_mask = (camera_gate_alignment > -0.707) & (camera_gate_alignment <= 0.0)  # cos(135°) ≈ -0.707
    camera_facing_reward[poor_mask] = 0.2 * parameter_dict["camera_facing_reward_magnitude"] * camera_gate_alignment[poor_mask]  # Small penalty
    
    # SEVERELY MISALIGNED: 135-180° (alignment ≤ -0.707) - Strong penalty
    severe_mask = camera_gate_alignment <= -0.707
    camera_facing_reward[severe_mask] = 2.0 * parameter_dict["camera_facing_reward_magnitude"] * camera_gate_alignment[severe_mask]  # Strong penalty
    
    # Reward for gate alignment (being in front of gate opening)
    gate_alignment_reward = torch.zeros_like(gate_distance)
    # Check if robot is roughly aligned with gate opening (Y direction)
    aligned_mask = torch.abs(robot_position[:, 0] - gate_position[:, 0]) < 1.5  # Within gate width
    gate_alignment_reward[aligned_mask] = parameter_dict["gate_alignment_reward_magnitude"]
    
    # Enhanced center alignment rewards for precise gate navigation
    gate_center_bonus = torch.zeros_like(gate_distance)
    # Distance from gate center in X direction (horizontal alignment)
    x_distance_from_center = torch.abs(robot_position[:, 0] - gate_position[:, 0])
    # Distance from gate center in Z direction (vertical alignment)  
    z_distance_from_center = torch.abs(robot_position[:, 2] - (gate_position[:, 2] + 1.2))  # Gate center height ~1.2m
    
    # Check if robot is very close to gate center (within 0.5m in both X and Z)
    center_aligned_mask = (x_distance_from_center < 0.5) & (z_distance_from_center < 0.3)
    gate_center_bonus[center_aligned_mask] = parameter_dict["gate_center_bonus_magnitude"]
    
    # Check for gate passage (crossing Y = 0 plane with proper alignment)
    just_passed_gate = (
        (robot_position[:, 1] > gate_position[:, 1]) &  # In front of gate
        (torch.abs(robot_position[:, 0] - gate_position[:, 0]) < 1.5) &  # Within gate width
        (robot_position[:, 2] > 0.2) & (robot_position[:, 2] < 2.2) &  # Within gate height
        (~gate_passed)  # Haven't passed before
    )
    
    # Check for center passage (more precise alignment)
    just_passed_center = (
        just_passed_gate &  # Basic passage requirement
        (x_distance_from_center < 0.5) &  # Centered horizontally
        (z_distance_from_center < 0.3)    # Centered vertically
    )
    
    gate_passage_reward = torch.zeros_like(gate_distance)
    gate_passage_reward[just_passed_gate] = parameter_dict["gate_passage_reward_magnitude"]
    
    # Extra bonus for center passage
    gate_center_passage_bonus = torch.zeros_like(gate_distance)
    gate_center_passage_bonus[just_passed_center] = parameter_dict["gate_center_passage_bonus_magnitude"]
    
    # Update gate passed status
    gate_passed = gate_passed | just_passed_gate

    # NEW: Altitude maintenance reward to encourage optimal gate-level flying
    optimal_altitude_min = 1.4  # meters
    optimal_altitude_max = 1.6  # meters  
    current_altitude = robot_position[:, 2]
    
    # Calculate distance from optimal altitude range
    altitude_error = torch.zeros_like(current_altitude)
    # Below optimal range
    below_range_mask = current_altitude < optimal_altitude_min
    altitude_error[below_range_mask] = optimal_altitude_min - current_altitude[below_range_mask]
    # Above optimal range  
    above_range_mask = current_altitude > optimal_altitude_max
    altitude_error[above_range_mask] = current_altitude[above_range_mask] - optimal_altitude_max
    # Within optimal range - no error
    
    # Exponential reward for being at optimal altitude
    altitude_maintenance_reward = exponential_reward_function(
        parameter_dict["altitude_maintenance_reward_magnitude"],
        parameter_dict["altitude_maintenance_reward_exponent"],
        altitude_error,
    )

    # Combined reward - NOW INCLUDING CAMERA FACING REWARD AND ALTITUDE MAINTENANCE
    reward = (
        MULTIPLICATION_FACTOR_REWARD
        * (
            pos_reward
            + very_close_to_goal_reward
            + getting_closer_reward
            + distance_from_goal_reward
            + gate_approach_reward
            + gate_alignment_reward
            + gate_passage_reward
            + gate_center_bonus
            + gate_center_passage_bonus
            + camera_facing_reward  # Camera facing reward
            + altitude_maintenance_reward  # NEW: Altitude maintenance reward
        )
        + total_action_penalty
    )

    # Apply collision penalties
    reward[:] = torch.where(
        crashes > 0,
        parameter_dict["collision_penalty"] * torch.ones_like(reward),
        reward,
    )
    
    return reward, crashes, camera_gate_alignment
