from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np
import os
import math

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

from aerial_gym.utils.vae.vae_image_encoder import VAEImageEncoder

import gymnasium as gym
from gym.spaces import Dict, Box

# Isaac Gym imports for static camera management
from isaacgym import gymapi, gymtorch

logger = CustomLogger("navigation_task_gate")

# VERSION: 2025.01.09_v2 - Fixed AttributeError curriculum_log_file in subprocesses
# Added robust curriculum logging with proper hasattr() checks

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
        
        # CONFIG VERIFICATION: Print key reward parameters to verify loading
        logger.warning("="*60)
        logger.warning("CONFIG VERIFICATION - REWARD PARAMETERS:")
        logger.warning(f"pos_reward_magnitude: {self.task_config.reward_parameters['pos_reward_magnitude']}")
        logger.warning(f"very_close_to_goal_reward_magnitude: {self.task_config.reward_parameters['very_close_to_goal_reward_magnitude']}")
        logger.warning(f"getting_closer_reward_multiplier: {self.task_config.reward_parameters['getting_closer_reward_multiplier']}")
        logger.warning(f"gate_approach_reward_magnitude: {self.task_config.reward_parameters['gate_approach_reward_magnitude']}")
        logger.warning(f"gate_passage_reward_magnitude: {self.task_config.reward_parameters['gate_passage_reward_magnitude']}")
        logger.warning(f"camera_facing_reward_magnitude: {self.task_config.reward_parameters.get('camera_facing_reward_magnitude', 'NOT FOUND!')}")
        logger.warning(f"collision_penalty: {self.task_config.reward_parameters['collision_penalty']}")
        logger.warning("="*60)
        
        logger.info("Building environment for gate navigation task.")
        logger.info(
            "Sim Name: {}, Env Name: {}, Robot Name: {}, Controller Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )

        # CRITICAL FIX: Set curriculum level and obstacle count BEFORE building environment
        # This ensures the asset manager gets the correct count from the start
        self.curriculum_level = self.task_config.curriculum.min_level
        obstacles_behind_gate = self.task_config.curriculum.get_obstacle_count_behind_gate(self.curriculum_level)
        
        # FIXED: Calculate visible assets: 1 visible gate + walls (6) + curriculum obstacles + robot (1)
        # NOTE: Even though 11 gate variants are loaded, only 1 will be visible at any time
        # The other 10 gates are hidden by moving them to (-1000, -1000, -1000)
        visible_gates = 0  # Only 1 gate visible at a time (others hidden by gate selection system)
        walls = 6  # 6 boundary walls
        robot = 0  # Robot is NOT part of env_asset_state_tensor (handled separately)
        fixed_assets_visible = visible_gates + walls  # = 7 visible fixed assets
        total_obstacles_in_env = fixed_assets_visible + obstacles_behind_gate
        
        logger.info(f"PRE-INIT: Setting curriculum level {self.curriculum_level} with {obstacles_behind_gate} curriculum obstacles")
        logger.info(f"PRE-INIT: Visible assets (env assets only): {visible_gates} gate + {walls} walls + {obstacles_behind_gate} curriculum = {total_obstacles_in_env} total")
        # logger.warning(f"[OBSTACLE_FIX] PRE-INIT: Level {self.curriculum_level} should spawn {obstacles_behind_gate} curriculum obstacles")
        logger.info(f"PRE-INIT: Total obstacle count for asset manager: {total_obstacles_in_env}")

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
        
        # Immediately select a random gate variant once after creation (safety)
        if hasattr(self.sim_env, 'apply_gate_variant_selection'):
            logger.warning("[GateVariant] Initial selection after build (one-time)")
            self.sim_env.apply_gate_variant_selection(env_ids=torch.arange(self.sim_env.num_envs, device=self.device))
        
        # CRITICAL FIX: Immediately update the environment's obstacle count after creation
        if hasattr(self.sim_env, 'global_tensor_dict'):
            self.sim_env.global_tensor_dict["num_obstacles_in_env"] = total_obstacles_in_env
            logger.info(f"POST-INIT: Updated global_tensor_dict with obstacle count: {total_obstacles_in_env}")

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

        # Gate-specific tracking and adaptive dimensions
        self.gate_position = torch.zeros((self.sim_env.num_envs, 3), device=self.device)
        self.gate_approach_distance = torch.zeros(self.sim_env.num_envs, device=self.device)
        
        # Gate dimensions for adaptive rewards (updated per environment based on selected gate)
        self.gate_width = torch.zeros((self.sim_env.num_envs,), device=self.device)  # Y-axis width
        self.gate_height = torch.zeros((self.sim_env.num_envs,), device=self.device)  # Z-axis height  
        self.gate_center_height = torch.zeros((self.sim_env.num_envs,), device=self.device)  # Center height for rewards
        
        # Gate scale factors for each environment (will be updated when gate is selected)
        self.gate_scale_factors = torch.ones((self.sim_env.num_envs,), device=self.device)

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
            self.static_image_latents = torch.zeros(
                (self.sim_env.num_envs, self.task_config.vae_config.latent_dims),  # Same latent dims
                device=self.device,
                requires_grad=False,
            )
        else:
            self.shared_vae_model = lambda x: x
            self.image_latents = torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            )
            self.static_image_latents = torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            )

        # Static camera management using Isaac Gym native API
        self.static_camera_manager = StaticCameraManager(self.sim_env, self.task_config)

        # Get the dictionary once from the environment and use it to get the observations later.
        self.obs_dict = self.sim_env.get_obs()
        
        # DEBUG: Print all available observations at startup for debugging
        logger.warning("="*80)
        logger.warning("🔍 AVAILABLE OBSERVATIONS AT INITIALIZATION:")
        for key, value in self.obs_dict.items():
            if hasattr(value, 'shape'):
                logger.warning(f"  📊 {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                logger.warning(f"  📊 {key}: {type(value)} = {value}")
        logger.warning("="*80)
        
        # IMMEDIATE DEBUG: Test observation processing right after initialization  
        logger.warning("🧪 TESTING OBSERVATION PROCESSING AT INITIALIZATION:")
        try:
            # Initialize task_obs first to avoid AttributeError
            if not hasattr(self, 'task_obs'):
                self.task_obs = {
                    "observations": torch.zeros(
                        (self.sim_env.num_envs, self.task_config.observation_space_dim),
                        device=self.device,
                        requires_grad=False,
                    ),
                }
            
            self.process_obs_for_task()
            logger.warning("✅ process_obs_for_task() executed successfully")
            
            # Check if task_obs was populated
            if hasattr(self, 'task_obs') and 'observations' in self.task_obs:
                obs_shape = self.task_obs['observations'].shape
                logger.warning(f"📊 task_obs shape: {obs_shape}")
                
                if obs_shape[1] == 150:  # 150D observation space
                    obs_sample = self.task_obs["observations"][0]  # First environment
                    
                    # Static camera data verification
                    static_cam_pos = obs_sample[3:6]
                    static_cam_orient = obs_sample[6:9] 
                    static_vae = obs_sample[86:150]
                    
                    logger.warning("🔍 IMMEDIATE STATIC CAMERA VERIFICATION:")
                    logger.warning(f"  📍 Static cam pos [3:6]: {static_cam_pos.cpu().numpy()}")
                    logger.warning(f"  📍 Static cam pos sum: {torch.sum(static_cam_pos).item():.6f}")
                    logger.warning(f"  🧭 Static cam orient [6:9]: {static_cam_orient.cpu().numpy()}")
                    logger.warning(f"  🧭 Static cam orient sum: {torch.sum(static_cam_orient).item():.6f}")
                    logger.warning(f"  📷 Static VAE sum: {torch.sum(static_vae).item():.6f}")
                    logger.warning(f"  📷 Static VAE norm: {torch.norm(static_vae).item():.6f}")
                    
                    # Critical checks
                    pos_is_zero = torch.allclose(static_cam_pos, torch.zeros_like(static_cam_pos), atol=1e-6)
                    orient_is_zero = torch.allclose(static_cam_orient, torch.zeros_like(static_cam_orient), atol=1e-6)
                    vae_is_zero = torch.allclose(static_vae, torch.zeros_like(static_vae), atol=1e-6)
                    
                    logger.warning(f"  ✅ Position populated: {'❌ ALL ZEROS' if pos_is_zero else '✅ NON-ZERO'}")
                    logger.warning(f"  ✅ Orientation populated: {'❌ ALL ZEROS' if orient_is_zero else '✅ NON-ZERO'}")
                    logger.warning(f"  ✅ VAE latents populated: {'❌ ALL ZEROS' if vae_is_zero else '✅ NON-ZERO'}")
                else:
                    logger.warning(f"❌ Wrong observation space dimension: {obs_shape[1]} (expected 150)")
            else:
                logger.warning("❌ task_obs not found or observations key missing")
                
        except Exception as e:
            logger.warning(f"❌ Error in process_obs_for_task(): {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
        
        logger.warning("="*80)
        
        # Use the curriculum level that was already set during pre-initialization
        self.obs_dict["curriculum_level"] = self.curriculum_level
        
        # Track maximum curriculum level reached (for no-decrease policy)
        self.max_curriculum_level_reached = self.curriculum_level
        
        # ===== CURRICULUM PARAMETER INITIALIZATION =====
        # Get obstacle count using the already-set curriculum level
        obstacles_behind_gate = self.task_config.curriculum.get_obstacle_count_behind_gate(self.curriculum_level)
        
        # Initialize curriculum logging first
        self.curriculum_log_file = None  # Initialize to None
        self.setup_curriculum_logging()  # Set up logging before using it
        
        # FIXED CALCULATION: Account for visible assets only (not all loaded assets)
        # Even though 11 gate variants are loaded, only 1 is visible at any time
        visible_gates = 1  # Only 1 gate visible at a time (others hidden by gate selection system)
        walls = 6  # 6 boundary walls  
        robot = 1  # 1 robot
        fixed_assets_visible = visible_gates + walls + robot  # = 8 visible fixed assets
        
        total_obstacles_in_env = fixed_assets_visible + obstacles_behind_gate
        
        logger.info(f"CURRICULUM: Visible assets: {visible_gates} gate + {walls} walls + {robot} robot + {obstacles_behind_gate} curriculum = {total_obstacles_in_env} total")
        logger.warning(f"[OBSTACLE_FIX] CURRICULUM: Level {self.curriculum_level} should spawn {obstacles_behind_gate} curriculum obstacles")
        
        # Update observation dictionary with obstacle count
        self.obs_dict["num_obstacles_in_env"] = total_obstacles_in_env
        logger.warning(f"[OBSTACLE_DEBUG] Task setting obs_dict num_obstacles_in_env = {total_obstacles_in_env}")
        
        # Confirm the environment manager has the correct count
        if hasattr(self.sim_env, 'global_tensor_dict'):
            old_count = self.sim_env.global_tensor_dict.get("num_obstacles_in_env", 0)
            if old_count != total_obstacles_in_env:
                logger.warning(f"[OBSTACLE_DEBUG] MISMATCH: Updating global_tensor_dict from {old_count} to {total_obstacles_in_env}")
                self.sim_env.global_tensor_dict["num_obstacles_in_env"] = total_obstacles_in_env
            else:
                logger.warning(f"[OBSTACLE_DEBUG] CONFIRMED: Global tensor dict already has correct obstacle count: {total_obstacles_in_env}")
        
        logger.info(f"FINAL: Visible assets: {fixed_assets_visible}, Curriculum obstacles: {obstacles_behind_gate}, Total: {total_obstacles_in_env}")
        
        # Initialize camera difficulty parameters (only static camera curriculum remains)
        self.max_camera_angle, self.camera_height_offset, self.camera_distance_offset = self.task_config.curriculum.get_static_camera_difficulty(self.curriculum_level)
        
        # ===== CURRICULUM LOGGING =====
        logger.info(f"INITIAL CURRICULUM (Level {self.curriculum_level}):")
        logger.info(f"   1. OBSTACLES: {obstacles_behind_gate} behind gate (total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)")
        try:
            sr = self.task_config.curriculum.get_spawn_ranges(self.curriculum_level)
            logger.info(
                f"   2. SPAWN: X∈[{(-sr['x_half_span_m']):.1f}, {(+sr['x_half_span_m']):.1f}] m, "
                f"Y∈[{(sr['y_center_m']-sr['y_half_span_m']):.1f}, {(sr['y_center_m']+sr['y_half_span_m']):.1f}] m, "
                f"Z∈[{(sr['z_center_m']-sr['z_half_span_m']):.1f}, {(sr['z_center_m']+sr['z_half_span_m']):.1f}] m; yaw ±{(sr['yaw_abs_rad']*57.2958):.1f}°"
            )
        except Exception as e:
            logger.info(f"   2. SPAWN: (fallback) Using fixed LMF2 config due to: {e}")
        logger.info(f"   3. CAMERA ANGLE: ±{self.max_camera_angle:.1f}deg max range (randomized per episode reset, fixed during episode)")
        
        # 5. CAMERA NOISE PROGRESSION (D455 Simulation)
        initial_camera_gaussian_std, initial_camera_dropout_rate = self.task_config.curriculum.get_camera_noise(self.curriculum_level)
        logger.info(f"   5. CAMERA NOISE: Gaussian STD={initial_camera_gaussian_std:.4f}, Dropout={initial_camera_dropout_rate*100:.1f}% (both drone & static)")
        
        # 6. CAMERA FRAME DROPOUT (entire-frame)
        fd = self.task_config.curriculum.get_camera_frame_dropout(self.curriculum_level)
        logger.info(f"   6. CAMERA FRAME DROPOUT: drone_total={fd['drone_total']*100:.1f}% (freeze {fd['drone_freeze']*100:.1f}%, blank {fd['drone_blank']*100:.1f}%), static_total={fd['static_total']*100:.1f}% (freeze {fd['static_freeze']*100:.1f}%, blank {fd['static_blank']*100:.1f}%)")
        
        # 7. STATE NOISE (pose) — new
        if getattr(self.task_config.curriculum, "enable_state_noise", False):
            sn = self.task_config.curriculum.get_state_noise(self.curriculum_level)
            logger.info(
                f"   7. STATE NOISE: drone_pos_std={sn['drone_pos_std_m']:.4f} m, drone_orient_std={sn['drone_orient_std_rad']*57.2958:.3f} deg, "
                f"static_pos_std={sn['static_pos_std_m']:.4f} m, static_orient_std={sn['static_orient_std_rad']*57.2958:.3f} deg"
            )
        else:
            logger.info("   7. STATE NOISE: disabled")
        
        logger.info(f"   8. ASSET MANAGER: Updated both obs_dict and global_tensor_dict with count {total_obstacles_in_env}")
        
        # Calculate progress fraction
        self.curriculum_progress_fraction = (
            self.curriculum_level - self.task_config.curriculum.min_level
        ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)
        
        logger.info(f"   8. PROGRESS: {self.curriculum_progress_fraction:.3f} (level {self.curriculum_level}/{self.task_config.curriculum.max_level})")
        logger.info(f"   9. EVALUATION: Check every {self.task_config.curriculum.check_after_log_instances} instances (success rate threshold: {self.task_config.curriculum.success_rate_for_increase:.3f})")
        
        self.log_curriculum_update(f"[INIT] Multi-aspect curriculum initialized at level {self.curriculum_level}")

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        # Enhanced observation space for gate navigation with static camera
        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.task_config.observation_space_dim,),  # 150D: 3D drone position + 6D static camera pose + 3D full orientation + 10D state + 64D drone VAE + 64D static camera VAE
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
        
        # Curriculum logging already initialized earlier in __init__

        self.pos_error_vehicle_frame = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.pos_error_vehicle_frame_prev = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.gate_passed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.camera_alignment_debug = torch.zeros(self.num_envs, device=self.device)
        self.num_task_steps = 0
        self.curriculum_progress_fraction = 0.0
        
        # EPISODE-LEVEL REWARD TRACKING: Track cumulative contributions per episode
        self.episode_pos_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_very_close_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_getting_closer_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_gate_approach_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_gate_alignment_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_camera_facing_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_action_penalty = torch.zeros(self.num_envs, device=self.device)
        self.episode_gate_passage_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_collision_penalty = torch.zeros(self.num_envs, device=self.device)
        self.episode_image_reward = torch.zeros(self.num_envs, device=self.device)
        
        # Track episode statistics
        self.episode_lengths = torch.zeros(self.num_envs, device=self.device)
        self.completed_episodes = []  # Store last 10 episode breakdowns
        self.max_stored_episodes = 10
        
        # Initialize gate dimensions for all environments after full initialization
        logger.warning("[GATE_ADAPTIVE] Initializing gate dimensions for all environments")
        self.update_gate_dimensions_for_environments(torch.arange(self.sim_env.num_envs, device=self.device))

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
        
        # Close curriculum log file
        if hasattr(self, 'curriculum_log_file') and self.curriculum_log_file:
            try:
                import datetime
                self.curriculum_log_file.write(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training session ended.\n")
                self.curriculum_log_file.write("="*80 + "\n")
                self.curriculum_log_file.close()
            except Exception as e:
                print(f"[DEBUG] Error closing curriculum log: {e}")

    def setup_curriculum_logging(self):
        """Setup separate curriculum logging file in train_dir."""
        try:
            # Try to determine train_dir path from Sample Factory environment or working directory
            import os
            import datetime
            
            # Get train_dir from environment variable or use current directory/train_dir
            train_dir = os.environ.get('SF_TRAIN_DIR', './train_dir')
            
            # If train_dir doesn't exist, create it
            os.makedirs(train_dir, exist_ok=True)
            
            # Create curriculum log filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            curriculum_log_filename = f"curriculum_gate_navigation_{timestamp}.log"
            curriculum_log_path = os.path.join(train_dir, curriculum_log_filename)
            
            # Open curriculum log file in UTF-8 encoding
            self.curriculum_log_file = open(curriculum_log_path, 'w', encoding='utf-8')
            
            # Log initial setup
            init_message = f"=== CURRICULUM LOGGING STARTED ===\nTimestamp: {timestamp}\nLog file: {curriculum_log_path}\n"
            self.curriculum_log_file.write(init_message)
            self.curriculum_log_file.flush()
            
            logger.info(f"Curriculum logging setup successful: {curriculum_log_path}")
            
        except Exception as e:
            # If curriculum logging setup fails, continue without it
            logger.warning(f"Failed to setup curriculum logging: {e}")
            logger.warning("Continuing without curriculum file logging (console logging still active)")
            self.curriculum_log_file = None

    def log_curriculum_update(self, message):
        """Log curriculum update messages to both console and curriculum log file."""
        try:
            # Always log to console
            logger.warning(message)
            
            # Try to log to file if available
            if hasattr(self, 'curriculum_log_file') and self.curriculum_log_file:
                try:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.curriculum_log_file.write(f"[{timestamp}] {message}\n")
                    self.curriculum_log_file.flush()  # Ensure immediate write
                except Exception as e:
                    # If file logging fails, continue without it
                    logger.debug(f"Failed to write to curriculum log file: {e}")
        except Exception as e:
            # If anything fails, just log to console
            logger.warning(f"Curriculum update: {message}")
            logger.debug(f"Curriculum logging error: {e}")

    def reset(self):
        self.reset_idx(torch.arange(self.sim_env.num_envs))
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        """
        SIMPLIFIED RESET WITH FIXED SPAWNING PARAMETERS
        
        Uses LMF2 robot config for spawning with fixed parameters:
        - ±0.5m lateral variation from gate center
        - ±45° orientation randomization  
        - Minimal initial velocity for randomization
        """
        # Set target positions (goals remain on front side of gate)
        target_ratio = torch_rand_float_tensor(self.target_min_ratio, self.target_max_ratio)
        self.target_position[env_ids] = torch_interpolate_ratio(
            min=self.obs_dict["env_bounds_min"][env_ids],
            max=self.obs_dict["env_bounds_max"][env_ids],
            ratio=target_ratio[env_ids],
        )
        
        # Robot spawning is now handled by the normal Isaac Gym reset mechanism
        # which uses the min_init_state and max_init_state from LMF2 config
        # This provides ±0.5m lateral variation and ±45° orientation automatically
        
        # Reset gate-specific tracking
        self.gate_passed[env_ids] = False
        self.gate_approach_distance[env_ids] = 0.0
        
        # RESET EPISODE REWARD TRACKING: Store completed episode data and reset trackers
        self.reset_episode_reward_tracking(env_ids)
        
        # Update static camera position based on curriculum level (ONLY for resetting environments)
        if len(env_ids) > 0:
            self.static_camera_manager.update_camera_positions(self.curriculum_level, env_ids)
            logger.debug(f"Updated static camera angles for {len(env_ids)} resetting environments: {env_ids.tolist()}")
        
        # Add debugging for gate randomization on reset
        # logger.warning(f"[GATE_RESET_DEBUG] Episode reset for environments {env_ids.tolist()} - new random gate sizes will be selected")
        
        # Persist curriculum level for env manager (for gated gate randomization)
        self.sim_env.global_tensor_dict["curriculum_level"] = int(self.curriculum_level)
        # Update gate dimensions for adaptive rewards after gate selection
        self.update_gate_dimensions_for_environments(env_ids)
        
        self.infos = {}
        return
    
    def extract_gate_dimensions_from_urdf(self, urdf_path):
        """
        Extract gate dimensions from URDF file.
        Returns (width, height, center_height, scale_factor)
        """
        import xml.etree.ElementTree as ET
        import os
        
        try:
            if not os.path.exists(urdf_path):
                logger.warning(f"[GATE_ADAPTIVE] URDF file not found: {urdf_path}, using default dimensions")
                return 2.5, 2.4, 1.2, 1.0  # Default 100% scale gate
            
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            
            # Extract scale factor from filename
            filename = os.path.basename(urdf_path)
            scale_factor = 1.0
            if "gate_scale_" in filename:
                try:
                    scale_str = filename.replace("gate_scale_", "").replace(".urdf", "")
                    scale_factor = int(scale_str) / 100.0
                except:
                    scale_factor = 1.0
            
            # Find left and right post positions to calculate width
            width = 2.5 * scale_factor  # Default scaled width
            height = 2.4 * scale_factor  # Default scaled height
            center_height = 1.2 * scale_factor  # Default scaled center height
            
            # Parse joint positions for more accurate dimensions
            for joint in root.iter('joint'):
                if joint.get('name') == 'base_to_left_post':
                    origin = joint.find('origin')
                    if origin is not None:
                        xyz = origin.get('xyz', '0 0 0').split()
                        left_y = abs(float(xyz[1]))
                        width = left_y * 2  # Total width = 2 * distance from center
                
                elif joint.get('name') == 'base_to_top_bar':
                    origin = joint.find('origin')
                    if origin is not None:
                        xyz = origin.get('xyz', '0 0 0').split()
                        top_z = float(xyz[2])
                        height = top_z  # Height to top bar
                        center_height = top_z / 2  # Center height
            
            # logger.warning(f"[GATE_ADAPTIVE] Extracted dimensions from {filename}: width={width:.3f}m, height={height:.3f}m, center_height={center_height:.3f}m, scale={scale_factor:.2f}")
            return width, height, center_height, scale_factor
            
        except Exception as e:
            logger.warning(f"[GATE_ADAPTIVE] Error parsing URDF {urdf_path}: {e}, using default dimensions")
            return 2.5, 2.4, 1.2, 1.0
    
    def calculate_gate_dimensions_from_name(self, gate_name):
        """
        Calculate gate dimensions from the gate name (e.g., gate_scale_060 -> 60% scale).
        Returns (width, height, center_height, scale_factor)
        """
        try:
            # Extract scale factor from gate name
            if "gate_scale_" in gate_name:
                scale_str = gate_name.replace("gate_scale_", "")
                scale_factor = int(scale_str) / 100.0
            else:
                scale_factor = 1.0
            
            # Base dimensions for 100% gate
            base_width = 2.5
            base_height = 2.4
            base_center_height = 1.2
            
            # Calculate scaled dimensions
            width = base_width * scale_factor
            height = base_height * scale_factor
            center_height = base_center_height * scale_factor
            
            logger.warning(f"[GATE_ADAPTIVE] Calculated dimensions from name '{gate_name}': width={width:.3f}m, height={height:.3f}m, center_height={center_height:.3f}m, scale={scale_factor:.2f}")
            return width, height, center_height, scale_factor
            
        except Exception as e:
            logger.warning(f"[GATE_ADAPTIVE] Error calculating dimensions from name '{gate_name}': {e}, using default")
            return 2.5, 2.4, 1.2, 1.0
    
    def update_gate_dimensions_for_environments(self, env_ids):
        """
        Update gate dimensions for specified environments based on their selected gate variants.
        """
        if not hasattr(self.sim_env, 'global_tensor_dict'):
            return
            
        # Safety check: ensure gate dimension attributes exist
        if not hasattr(self, 'gate_width') or not hasattr(self, 'gate_height'):
            logger.warning("[GATE_ADAPTIVE] Gate dimension attributes not initialized yet, skipping update")
            return
            
        gate_variant_names = self.sim_env.global_tensor_dict.get("gate_variant_names_per_env", [])
        active_gate_array_indices = self.sim_env.global_tensor_dict.get("active_gate_variant_array_index", torch.zeros(self.sim_env.num_envs))
        
        for env_id in (env_ids.tolist() if hasattr(env_ids, 'tolist') else [env_ids]):
            if env_id >= len(gate_variant_names):
                continue
                
            env_gate_names = gate_variant_names[env_id]
            active_idx = active_gate_array_indices[env_id].item()
            
            if active_idx >= 0 and active_idx < len(env_gate_names):
                # Get the active gate variant name
                active_gate_name = env_gate_names[active_idx] if env_gate_names else "gate_scale_100"
                
                # Construct URDF path - find the correct base directory
                urdf_filename = f"{active_gate_name}.urdf"
                
                # Try multiple possible base directories to find the URDF files
                possible_base_dirs = [
                    os.getcwd(),  # Current working directory
                    os.path.dirname(os.path.abspath(__file__)),  # Directory of this file
                    "/home/ziyar/aerialgym/aerialgym_ws/src/aerial_gym_simulator",  # Known project root
                ]
                
                # Add parent directories up to 5 levels
                current_dir = os.getcwd()
                for _ in range(5):
                    current_dir = os.path.dirname(current_dir)
                    possible_base_dirs.append(current_dir)
                
                urdf_path = None
                for base_dir in possible_base_dirs:
                    test_path = os.path.join(base_dir, "resources/models/environment_assets/gates", urdf_filename)
                    if os.path.exists(test_path):
                        urdf_path = test_path
                        break
                
                if urdf_path is None:
                    # Fallback: construct path anyway for the error message
                    urdf_path = os.path.join(possible_base_dirs[0], "resources/models/environment_assets/gates", urdf_filename)
                
                # Extract dimensions from URDF or calculate from filename
                if urdf_path and os.path.exists(urdf_path):
                    width, height, center_height, scale_factor = self.extract_gate_dimensions_from_urdf(urdf_path)
                else:
                    # Fallback: calculate dimensions from scale factor in filename
                    width, height, center_height, scale_factor = self.calculate_gate_dimensions_from_name(active_gate_name)
                
                # Update environment-specific dimensions
                self.gate_width[env_id] = width
                self.gate_height[env_id] = height
                self.gate_center_height[env_id] = center_height
                self.gate_scale_factors[env_id] = scale_factor
                
                # logger.warning(f"[GATE_ADAPTIVE] Env {env_id}: Updated to gate '{active_gate_name}' - width={width:.3f}m, height={height:.3f}m, scale={scale_factor:.2f}")
            else:
                # Default dimensions if no active gate found
                self.gate_width[env_id] = 2.5
                self.gate_height[env_id] = 2.4
                self.gate_center_height[env_id] = 1.2
                self.gate_scale_factors[env_id] = 1.0
                logger.warning(f"[GATE_ADAPTIVE] Env {env_id}: No active gate found (active_idx={active_idx}, num_gates={len(env_gate_names)}), using default gate dimensions")
    
    # REMOVED: _apply_curriculum_drone_spawning and _apply_curriculum_orientation_randomization
    # These methods have been removed as we now use fixed parameters from LMF2 config
    # The normal Isaac Gym reset mechanism handles spawning using min_init_state/max_init_state

    def render(self):
        return self.sim_env.render()

    def step(self, actions):
        # VELOCITY CONTROLLER: Transform 4D actions to direct velocity commands for LMF2 robot
        # Input: [x_vel_cmd, y_vel_cmd, z_vel_cmd, yaw_rate_cmd] ∈ [-1, 1]^4
        # Output: [x_vel, y_vel, z_vel, yaw_rate] applied directly as velocity commands
        
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

        # ===== SIMPLE GATE PASSAGE SUCCESS CRITERIA =====
        # Success = simply passing through the gate boundary (any part of the gate opening)
        # More forgiving than target-based or centered passage requirements
        robot_position = self.obs_dict["robot_position"]
        
        # Gate passage detection: crossed gate plane with proper alignment - ADAPTIVE to gate dimensions
        # Gate dimensions are now adaptive based on the selected gate variant per environment
        gate_success_width_tolerance = self.gate_width * 0.52  # 52% of gate width for success detection (safety margin)
        gate_success_min_height = self.gate_position[:, 2] + self.gate_height * 0.08  # 8% above ground
        gate_success_max_height = self.gate_position[:, 2] + self.gate_height * 0.92  # 92% of gate height
        
        gate_passage_success = (
            (robot_position[:, 1] > self.gate_position[:, 1]) &  # Crossed gate (Y > 0)
            (torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < gate_success_width_tolerance) &  # Within gate width
            (robot_position[:, 2] > gate_success_min_height) & (robot_position[:, 2] < gate_success_max_height)  # Within gate height range
        )
        
        # Success when episode truncates (not crashes) and gate passage achieved
        successes = self.truncations * gate_passage_success
        successes = torch.where(self.terminations > 0, torch.zeros_like(successes), successes)
        
        # Keep original target-based success as backup for comparison
        target_success = torch.norm(self.target_position - robot_position, dim=1) < 1.0
        target_successes = self.truncations * target_success
        target_successes = torch.where(self.terminations > 0, torch.zeros_like(target_successes), target_successes)
        
        # Use gate passage as primary, but also accept target success if achieved
        successes = torch.logical_or(successes, target_successes)
        

        # ===== END SIMPLE GATE PASSAGE SUCCESS =====
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
        
        # Check if robot has passed gate (crossed Y = 0 plane with proper alignment) - ADAPTIVE
        gate_tracking_width_tolerance = self.gate_width * 0.6  # 60% of gate width for tracking
        gate_tracking_min_height = self.gate_position[:, 2] + self.gate_height * 0.08  # 8% above ground
        gate_tracking_max_height = self.gate_position[:, 2] + self.gate_height * 0.92  # 92% of gate height
        
        gate_passed_current = (
            (robot_position[:, 1] > self.gate_position[:, 1]) &  # In front of gate
            (torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < gate_tracking_width_tolerance) &  # Within gate width
            (robot_position[:, 2] > gate_tracking_min_height) & (robot_position[:, 2] < gate_tracking_max_height)  # Within gate height
        )
        
        # Gate alignment: check if robot is roughly aligned with gate opening - ADAPTIVE
        gate_alignment = torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < gate_tracking_width_tolerance
        
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
        
        # FINAL VERIFICATION: After all processing is complete
        if not hasattr(self, '_final_verification_printed'):
            self._final_verification_printed = True
            logger.warning("🎯 FINAL STATIC CAMERA VERIFICATION (AFTER PROCESSING):")
            
            # Process observations to get final state
            self.process_obs_for_task()
            
            if hasattr(self, 'task_obs') and 'observations' in self.task_obs:
                obs_sample = self.task_obs["observations"][0]
                
                static_pos = obs_sample[3:6]
                static_orient = obs_sample[6:9]
                static_vae = obs_sample[86:150]
                
                logger.warning(f"  📍 Final static pos: {static_pos.cpu().numpy()}")
                logger.warning(f"  🧭 Final static orient: {static_orient.cpu().numpy()}")
                logger.warning(f"  📷 Final static VAE: range=[{static_vae.min().item():.3f}, {static_vae.max().item():.3f}]")
                
                # Check final state
                pos_ok = not torch.allclose(static_pos, torch.zeros_like(static_pos), atol=1e-6)
                orient_ok = not torch.allclose(static_orient, torch.zeros_like(static_orient), atol=1e-6)
                vae_ok = not torch.allclose(static_vae, torch.zeros_like(static_vae), atol=1e-6)
                
                logger.warning(f"  ✅ FINAL RESULTS: pos={pos_ok}, orient={orient_ok}, vae={vae_ok}")
                
                if pos_ok and orient_ok and vae_ok:
                    logger.warning("🎉 SUCCESS: All 150D static camera observations verified!")
                    
                    # CRITICAL: Add verification that observations reach RL training
                    logger.warning("🤖 RL TRAINING USAGE VERIFICATION:")
                    logger.warning("  ⚠️  IMPORTANT: This verifies DATA PIPELINE, not RL training usage!")
                    logger.warning("  📋 To verify RL training usage, check:")
                    logger.warning("     1. Neural network receives 150D input (not 128D or other)")
                    logger.warning("     2. Policy network architecture matches observation space")
                    logger.warning("     3. Static camera indices [3:6] and [86:150] affect policy decisions")
                    logger.warning("     4. Ablation test: performance difference with vs without static camera")
                    logger.warning("  🔍 Current verification: Environment correctly provides 150D observations")
                    logger.warning("  ❓ Next step needed: Verify Sample Factory & neural network usage")
                else:
                    logger.error("❌ Some static camera data still missing after processing!")
        
        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()
        return return_tuple

    def process_image_observation(self):
        """Process drone camera observations with D455 curriculum-dependent noise."""
        # Get the drone's depth image (normalized 0.0–1.0)
        image_obs = self.obs_dict["depth_range_pixels"].squeeze(1)  # shape: (num_envs, H, W)
        
        # Apply D455 camera noise if enabled and at sufficient curriculum level
        noised_image_obs = image_obs.clone()  # Start with clean image
        if getattr(self.task_config.curriculum, "enable_camera_noise", False):
            # Get noise parameters for current level
            gaussian_std, dropout_rate = self.task_config.curriculum.get_camera_noise(self.curriculum_level)
            if gaussian_std > 0 or dropout_rate > 0:
                # Gaussian noise: add N(0, gaussian_std) to each pixel (depth measurement uncertainty)
                noise = torch.randn_like(noised_image_obs) * gaussian_std
                noised_image_obs = noised_image_obs + noise
                
                # Pixel dropout: set a fraction of pixels to 1.0 (missing depth readings)
                if dropout_rate > 0:
                    dropout_mask = torch.rand_like(noised_image_obs) < dropout_rate
                    noised_image_obs = noised_image_obs.masked_fill(dropout_mask, 1.0)  # 1.0 = max depth (no reading)
                
                # Clamp values to valid range [0, 1]
                noised_image_obs = torch.clamp(noised_image_obs, 0.0, 1.0)
        
        # Entire-frame dropout (curriculum-driven)
        if getattr(self.task_config.curriculum, "enable_camera_frame_dropout", False):
            fd = self.task_config.curriculum.get_camera_frame_dropout(self.curriculum_level)
            p_blank = fd.get("drone_blank", 0.0)
            p_freeze = fd.get("drone_freeze", 0.0)
            if (p_blank > 0.0) or (p_freeze > 0.0):
                # Ensure buffer exists
                if not hasattr(self, "_prev_drone_depth"):
                    self._prev_drone_depth = noised_image_obs.clone()
                # Draw per-env Bernoulli masks
                if p_blank > 0.0:
                    blank_mask = (torch.rand(noised_image_obs.shape[0], device=noised_image_obs.device) < p_blank).view(-1, 1, 1)
                    noised_image_obs = torch.where(blank_mask, torch.ones_like(noised_image_obs), noised_image_obs)
                if p_freeze > 0.0:
                    freeze_mask = (torch.rand(noised_image_obs.shape[0], device=noised_image_obs.device) < p_freeze).view(-1, 1, 1)
                    # Apply freeze only where not already blanked
                    apply_freeze = freeze_mask if p_blank == 0.0 else (freeze_mask & (~blank_mask))
                    noised_image_obs = torch.where(apply_freeze, self._prev_drone_depth, noised_image_obs)
                # Update previous buffer after potential dropout
                self._prev_drone_depth = noised_image_obs.clone()
        else:
            # Maintain previous buffer if feature disabled
            if not hasattr(self, "_prev_drone_depth"):
                self._prev_drone_depth = noised_image_obs.clone()
            else:
                self._prev_drone_depth = noised_image_obs.clone()
        
        # Store noised drone camera image for GIF generation (add channel dimension back)
        self.obs_dict["depth_range_pixels_noised"] = noised_image_obs.unsqueeze(1)  # shape: (num_envs, 1, H, W)
        
        # Encode the (potentially noisy) image using VAE
        if self.task_config.vae_config.use_vae:
            self.image_latents[:] = self.shared_vae_model.encode(noised_image_obs)

    def process_static_camera_observation(self):
        """Process static camera observations with D455 curriculum-dependent noise."""
        try:
            static_depth, static_seg = self.static_camera_manager.capture_images()
            
            # CRITICAL DEBUG: Log static camera capture success/failure
            if not hasattr(self, '_static_debug_logged'):
                self._static_debug_logged = True
                if static_depth is not None:
                    logger.warning(f"✅ Static camera capture successful: shape={static_depth.shape if hasattr(static_depth, 'shape') else 'N/A'}, type={type(static_depth)}")
                else:
                    logger.warning("❌ Static camera capture failed: static_depth is None")
            
            if static_depth is not None and self.task_config.vae_config.use_vae:
                # Store clean static camera image (original)
                static_depth_clean = static_depth.copy() if isinstance(static_depth, np.ndarray) else static_depth.clone()
                
                # Apply D455 camera noise if enabled and at sufficient curriculum level
                static_depth_noised = static_depth_clean.copy() if isinstance(static_depth_clean, np.ndarray) else static_depth_clean.clone()
                if getattr(self.task_config.curriculum, "enable_camera_noise", False):
                    # Get noise parameters for current level
                    gaussian_std, dropout_rate = self.task_config.curriculum.get_camera_noise(self.curriculum_level)
                    if gaussian_std > 0 or dropout_rate > 0:
                        # Handle numpy array case
                        if isinstance(static_depth_noised, np.ndarray):
                            # Gaussian noise: add N(0, gaussian_std) to each pixel (depth measurement uncertainty)
                            noise = np.random.normal(0.0, gaussian_std, size=static_depth_noised.shape)
                            static_depth_noised = static_depth_noised + noise
                            
                            # Pixel dropout: set a fraction of pixels to 1.0 (missing depth readings)
                            if dropout_rate > 0:
                                dropout_mask = np.random.rand(*static_depth_noised.shape) < dropout_rate
                                static_depth_noised[dropout_mask] = 1.0  # 1.0 = max depth (no reading)
                            
                            # Clip depth values to [0, 1] range
                            static_depth_noised = np.clip(static_depth_noised, 0.0, 1.0)
                        else:
                            # Handle tensor case
                            noise = torch.randn_like(static_depth_noised) * gaussian_std
                            static_depth_noised = static_depth_noised + noise
                            
                            if dropout_rate > 0:
                                dropout_mask = torch.rand_like(static_depth_noised) < dropout_rate
                                static_depth_noised = static_depth_noised.masked_fill(dropout_mask, 1.0)
                            
                            static_depth_noised = torch.clamp(static_depth_noised, 0.0, 1.0)

                # Entire-frame dropout (curriculum-driven)
                if getattr(self.task_config.curriculum, "enable_camera_frame_dropout", False):
                    fd = self.task_config.curriculum.get_camera_frame_dropout(self.curriculum_level)
                    p_blank = fd.get("static_blank", 0.0)
                    p_freeze = fd.get("static_freeze", 0.0)
                    if (p_blank > 0.0) or (p_freeze > 0.0):
                        # Initialize previous static buffer
                        if not hasattr(self, "_prev_static_depth"):
                            # Use per-env shared static image
                            if isinstance(static_depth_noised, np.ndarray):
                                self._prev_static_depth = static_depth_noised.copy()
                            else:
                                self._prev_static_depth = static_depth_noised.clone()
                        # Apply blank then freeze
                        if isinstance(static_depth_noised, np.ndarray):
                            if p_blank > 0.0 and (np.random.rand() < p_blank):
                                static_depth_noised[...] = 1.0
                            elif p_freeze > 0.0 and (np.random.rand() < p_freeze):
                                static_depth_noised = self._prev_static_depth.copy()
                        else:
                            # tensor case
                            do_blank = (torch.rand(1, device=static_depth_noised.device).item() < p_blank)
                            if do_blank:
                                static_depth_noised = torch.ones_like(static_depth_noised)
                            else:
                                do_freeze = (torch.rand(1, device=static_depth_noised.device).item() < p_freeze)
                                if do_freeze:
                                    static_depth_noised = self._prev_static_depth.clone()
                        # update buffer
                        if isinstance(static_depth_noised, np.ndarray):
                            self._prev_static_depth = static_depth_noised.copy()
                        else:
                            self._prev_static_depth = static_depth_noised.clone()
                
                # Store noised static camera images for GIF generation
                self.obs_dict["static_depth_clean"] = static_depth_clean
                self.obs_dict["static_depth_noised"] = static_depth_noised
                self.obs_dict["static_seg"] = static_seg
                
                # CRITICAL FIX: Enhanced VAE encoding with detailed debugging
                try:
                    # Convert to tensor and process through VAE (use noised version for training)
                    if isinstance(static_depth_noised, np.ndarray):
                        static_depth_tensor = torch.from_numpy(static_depth_noised).float().to(self.device)
                        if static_depth_tensor.dim() == 2:
                            static_depth_tensor = static_depth_tensor.unsqueeze(0)  # Add batch dimension
                        # Ensure all environments get the same static camera view
                        static_depth_expanded = static_depth_tensor.expand(self.sim_env.num_envs, -1, -1)
                        
                        # CRITICAL DEBUG: Log VAE encoding attempt
                        if not hasattr(self, '_vae_debug_logged'):
                            self._vae_debug_logged = True
                            logger.warning(f"🔧 VAE encoding static camera: input_shape={static_depth_expanded.shape}, device={static_depth_expanded.device}")
                        
                        encoded_latents = self.shared_vae_model.encode(static_depth_expanded)
                        self.static_image_latents[:] = encoded_latents
                        
                        # CRITICAL DEBUG: Verify VAE output
                        if not hasattr(self, '_vae_output_logged'):
                            self._vae_output_logged = True
                            logger.warning(f"✅ VAE encoding successful: output_shape={encoded_latents.shape}, range=[{encoded_latents.min().item():.3f}, {encoded_latents.max().item():.3f}]")
                        
                    else:
                        # Direct tensor path with per-env identical static image
                        static_depth_tensor = static_depth_noised
                        if static_depth_tensor.dim() == 2:
                            static_depth_tensor = static_depth_tensor.unsqueeze(0)
                        static_depth_expanded = static_depth_tensor.expand(self.sim_env.num_envs, -1, -1)
                        encoded_latents = self.shared_vae_model.encode(static_depth_expanded)
                        self.static_image_latents[:] = encoded_latents
                except Exception as e:
                    logger.warning(f"VAE encoding of static camera failed: {e}")
            else:
                # No static camera data or VAE disabled
                if not hasattr(self, '_no_static_logged'):
                    self._no_static_logged = True
                    if static_depth is None:
                        logger.warning("❌ Static camera data is None - camera capture failed")
                    elif not self.task_config.vae_config.use_vae:
                        logger.warning("❌ VAE disabled in config - static camera latents will be zeros")
                
                # Fill with zeros if no data
                self.static_image_latents.fill_(0.0)
                
        except Exception as e:
            logger.error(f"❌ Static camera processing error: {e}")
            # Fallback to zeros on any error
            self.static_image_latents.fill_(0.0)

    def post_image_reward_addition(self):
        """Add image-based rewards from drone camera."""
        image_obs = 10.0 * self.obs_dict["depth_range_pixels"].squeeze(1)
        image_obs[image_obs < 0] = 10.0
        self.min_pixel_dist = torch.amin(image_obs, dim=(1, 2))
        
        # Calculate image rewards for debugging
        image_rewards = -exponential_reward_function(
            4.0, 1.0, self.min_pixel_dist[self.terminations < 0]
        )
        
        # COMPREHENSIVE IMAGE REWARD DEBUGGING: Print values every 200 steps  
        if hasattr(self, 'num_task_steps') and self.num_task_steps % 200 == 0:
            avg_min_dist = torch.mean(self.min_pixel_dist).item()
            avg_image_reward = torch.mean(image_rewards).item() if len(image_rewards) > 0 else 0.0
            min_pixel_dist = torch.min(self.min_pixel_dist).item()
            max_pixel_dist = torch.max(self.min_pixel_dist).item()
            
            # Count environments with different distance ranges
            very_close_count = torch.sum(self.min_pixel_dist < 2.0).item()  # < 2m
            close_count = torch.sum((self.min_pixel_dist >= 2.0) & (self.min_pixel_dist < 4.0)).item()  # 2-4m
            safe_count = torch.sum(self.min_pixel_dist >= 4.0).item()  # > 4m
            
            # COMMENTED OUT: Verbose image reward analysis (clutters training output)
            # logger.warning("="*60)
            # logger.warning(f"📷 IMAGE REWARD ANALYSIS (Step {self.num_task_steps}):")
            # logger.warning(f"  🎯 Average Image Reward:   {avg_image_reward:.3f}")
            # logger.warning(f"  📏 Distance Stats:")
            # logger.warning(f"    • Average:               {avg_min_dist:.2f}m")
            # logger.warning(f"    • Range:                 {min_pixel_dist:.2f}m - {max_pixel_dist:.2f}m")
            # logger.warning(f"  🚦 Environment Distribution:")
            # logger.warning(f"    • Very Close (<2m):      {very_close_count}/16 envs")
            # logger.warning(f"    • Close (2-4m):          {close_count}/16 envs")
            # logger.warning(f"    • Safe (>4m):            {safe_count}/16 envs")
            # 
            # # Safety warnings
            # if avg_min_dist < 1.5:
            #     logger.warning("  ⚠️  WARNING: Drones flying very close to obstacles!")
            # elif avg_image_reward < -2.0:
            #     logger.warning("  ⚠️  WARNING: High image penalties - collision avoidance active!")
            # elif very_close_count > 8:
            #     logger.warning("  ⚠️  WARNING: Many drones in danger zone (<2m from obstacles)!")
            # else:
            #     logger.warning("  ✅ Image rewards normal - good collision avoidance")
            # 
            # logger.warning("="*60)
        
        # Apply the image rewards
        self.rewards[self.terminations < 0] += image_rewards

    def get_return_tuple(self):
        self.process_obs_for_task()
        
        # ADDITIONAL DEBUG: Verify observations in get_return_tuple (called every step)
        if not hasattr(self, '_return_tuple_debug_printed'):
            self._return_tuple_debug_printed = True
            logger.warning("🎯 OBSERVATION VERIFICATION IN get_return_tuple():")
            
            if hasattr(self, 'task_obs') and 'observations' in self.task_obs:
                obs_shape = self.task_obs['observations'].shape
                logger.warning(f"📊 Final task_obs shape: {obs_shape}")
                
                if obs_shape[1] >= 150:
                    obs_sample = self.task_obs["observations"][0]
                    
                    # Core verification of static camera data
                    static_pos = obs_sample[3:6]
                    static_orient = obs_sample[6:9]
                    static_vae = obs_sample[86:150] if obs_shape[1] >= 150 else obs_sample[86:]
                    
                    logger.warning(f"🔍 FINAL VERIFICATION:")
                    logger.warning(f"  Static pos: {static_pos.cpu().numpy()}")
                    logger.warning(f"  Static orient: {static_orient.cpu().numpy()}")
                    logger.warning(f"  Static VAE range: [{static_vae.min().item():.3f}, {static_vae.max().item():.3f}]")
                    
                    # Check if properly populated
                    pos_nonzero = not torch.allclose(static_pos, torch.zeros_like(static_pos), atol=1e-6)
                    orient_nonzero = not torch.allclose(static_orient, torch.zeros_like(static_orient), atol=1e-6) 
                    vae_nonzero = not torch.allclose(static_vae, torch.zeros_like(static_vae), atol=1e-6)
                    
                    logger.warning(f"  ✅ Static camera working: pos={pos_nonzero}, orient={orient_nonzero}, vae={vae_nonzero}")
                    
                    if not (pos_nonzero and orient_nonzero and vae_nonzero):
                        logger.error("❌ CRITICAL: Static camera data is missing!")
                    else:
                        logger.warning("✅ SUCCESS: All static camera data populated correctly!")
                else:
                    logger.error(f"❌ WRONG OBSERVATION DIMENSION: {obs_shape[1]} (expected 150)")
            else:
                logger.error("❌ task_obs not available in get_return_tuple")
        
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        """
        Process observations for the gate navigation task.
        
        UPDATED: Now matches DCE navigation task format with static camera pose observations.
        
        Observation space (150D):
        - 0-3: Drone absolute position in world coordinates
        - 3-6: Static camera position relative to drone  
        - 6-9: Static camera orientation relative to drone
        - 9-12: Full drone orientation including yaw
        - 12-15: Robot body linear velocity
        - 15-18: Robot body angular velocity
        - 18-22: Robot actions (4D for gate navigation)
        - 22-86: Drone camera VAE latents (64D)
        - 86-150: Static camera VAE latents (64D)
        """
        # MODIFIED: Include drone absolute position and full orientation sensing
        # This provides the agent with complete spatial awareness of its state and static camera relative position
        
        # ===== DRONE ABSOLUTE POSITION OBSERVATIONS (3D) =====
        # [0:3] = Drone absolute position in world coordinates (x, y, z)
        drone_pos_clean = self.obs_dict["robot_position"]
        # Apply curriculum-driven state noise (drone position)
        if getattr(self.task_config.curriculum, "enable_state_noise", False):
            noise_cfg = self.task_config.curriculum.get_state_noise(self.curriculum_level)
            dp_std = float(noise_cfg.get("drone_pos_std_m", 0.0))
            if dp_std > 0.0:
                drone_pos_noised = drone_pos_clean + torch.randn_like(drone_pos_clean) * dp_std
            else:
                drone_pos_noised = drone_pos_clean
        else:
            drone_pos_noised = drone_pos_clean
        self.task_obs["observations"][:, 0:3] = drone_pos_noised
        
        # ===== STATIC CAMERA POSE OBSERVATIONS (6D) =====
        # Get static camera pose information relative to drone
        static_camera_pos, static_camera_orientation = self._get_static_camera_pose_relative_to_drone()
        # Apply curriculum-driven noise to static camera pose copies (not altering sim)
        if getattr(self.task_config.curriculum, "enable_state_noise", False):
            noise_cfg = self.task_config.curriculum.get_state_noise(self.curriculum_level)
            sp_std = float(noise_cfg.get("static_pos_std_m", 0.0))
            so_std = float(noise_cfg.get("static_orient_std_rad", 0.0))
            if sp_std > 0.0:
                static_camera_pos = static_camera_pos + torch.randn_like(static_camera_pos) * sp_std
            if so_std > 0.0:
                static_camera_orientation = static_camera_orientation + torch.randn_like(static_camera_orientation) * so_std
                # Wrap yaw-ish components into [-pi, pi] if needed (approx, for stability)
                static_camera_orientation = torch.atan2(torch.sin(static_camera_orientation), torch.cos(static_camera_orientation))
        
        # [3:6] = Static camera position relative to drone (x, y, z in drone's reference frame)
        self.task_obs["observations"][:, 3:6] = static_camera_pos
        
        # [6:9] = Static camera orientation relative to drone (roll, pitch, yaw in drone's reference frame)
        self.task_obs["observations"][:, 6:9] = static_camera_orientation
        
        # ===== DRONE FULL ORIENTATION OBSERVATIONS (3D) =====
        # [9:12] = Full drone orientation including yaw (roll, pitch, yaw)
        euler_angles = ssa(get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"]))
        # Apply curriculum-driven noise to drone orientation copy
        if getattr(self.task_config.curriculum, "enable_state_noise", False):
            noise_cfg = self.task_config.curriculum.get_state_noise(self.curriculum_level)
            do_std = float(noise_cfg.get("drone_orient_std_rad", 0.0))
            if do_std > 0.0:
                euler_angles = euler_angles + torch.randn_like(euler_angles) * do_std
                euler_angles = torch.atan2(torch.sin(euler_angles), torch.cos(euler_angles))
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

    def compute_rewards_and_crashes(self, obs_dict):
        """Compute rewards with gate-specific components."""
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        
        self.pos_error_vehicle_frame_prev[:] = self.pos_error_vehicle_frame
        self.pos_error_vehicle_frame[:] = quat_rotate_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        
        # CRITICAL FIX: Clone action tensors to break reference dependency
        # obs_dict contains direct references to global tensors that get updated simultaneously
        current_actions = obs_dict["robot_actions"].clone()
        previous_actions = obs_dict["robot_prev_actions"].clone()
        
        rewards, crashes, camera_gate_alignment = compute_gate_reward(
            self.pos_error_vehicle_frame,
            self.pos_error_vehicle_frame_prev,
            obs_dict["crashes"],
            current_actions,
            previous_actions,
            robot_position,
            robot_vehicle_orientation,
            self.gate_position,
            self.gate_passed,
            self.curriculum_progress_fraction,
            self.task_config.reward_parameters,
            self.gate_width,
            self.gate_height,
            self.gate_center_height,
        )
        
        # UPDATE EPISODE REWARD TRACKING: Track cumulative reward components
        self.update_episode_reward_tracking(obs_dict, rewards, crashes)
        
        # COMPREHENSIVE REWARD DEBUGGING: Print ALL reward components every 200 steps
        if hasattr(self, 'num_task_steps') and self.num_task_steps % 200 == 0:
            # Recalculate components for debugging (without JIT optimization)
            dist = torch.norm(self.pos_error_vehicle_frame, dim=1)
            prev_dist = torch.norm(self.pos_error_vehicle_frame_prev, dim=1)
            action = obs_dict["robot_actions"]
            prev_action = obs_dict["robot_prev_actions"]
            robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
            
            # Individual reward components (average across environments)
            pos_reward = exponential_reward_function(
                self.task_config.reward_parameters["pos_reward_magnitude"],
                self.task_config.reward_parameters["pos_reward_exponent"],
                dist,
            )
            
            very_close_reward = exponential_reward_function(
                self.task_config.reward_parameters["very_close_to_goal_reward_magnitude"],
                self.task_config.reward_parameters["very_close_to_goal_reward_exponent"],
                dist,
            )
            
            getting_closer = prev_dist - dist
            getting_closer_reward = torch.where(
                getting_closer > 0,
                self.task_config.reward_parameters["getting_closer_reward_multiplier"] * getting_closer,
                2.0 * self.task_config.reward_parameters["getting_closer_reward_multiplier"] * getting_closer,
            )
            
            gate_distance = torch.norm(robot_position - self.gate_position, dim=1)
            gate_approach_reward = exponential_reward_function(
                self.task_config.reward_parameters["gate_approach_reward_magnitude"],
                0.5,
                gate_distance,
            )
            
            # Gate alignment
            gate_alignment_reward = torch.zeros_like(gate_distance)
            aligned_mask = torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < 1.5
            gate_alignment_reward[aligned_mask] = self.task_config.reward_parameters["gate_alignment_reward_magnitude"]
            
            # Camera facing reward calculation (same as in compute_gate_reward)
            drone_to_gate = self.gate_position - robot_position
            drone_to_gate_normalized = drone_to_gate / (torch.norm(drone_to_gate, dim=1, keepdim=True) + 1e-8)
            
            # Get drone's forward direction (where camera points)
            qw, qx, qy, qz = robot_vehicle_orientation[:, 3], robot_vehicle_orientation[:, 0], robot_vehicle_orientation[:, 1], robot_vehicle_orientation[:, 2]
            forward_x = 1.0 - 2.0 * (qy * qy + qz * qz)
            forward_y = 2.0 * (qx * qy + qw * qz)
            forward_z = 2.0 * (qx * qz - qw * qy)
            drone_forward = torch.stack([forward_x, forward_y, forward_z], dim=1)
            drone_forward_normalized = drone_forward / (torch.norm(drone_forward, dim=1, keepdim=True) + 1e-8)
            
            # Calculate alignment between camera direction and gate direction
            camera_gate_alignment = torch.sum(drone_forward_normalized * drone_to_gate_normalized, dim=1)
            camera_gate_alignment = torch.clamp(camera_gate_alignment, -1.0, 1.0)
            
            # Camera facing reward with same logic as compute_gate_reward
            camera_facing_reward = torch.zeros_like(camera_gate_alignment)
            perfect_mask = camera_gate_alignment > 0.966
            camera_facing_reward[perfect_mask] = self.task_config.reward_parameters["camera_facing_reward_magnitude"]
            excellent_mask = (camera_gate_alignment > 0.866) & (camera_gate_alignment <= 0.966)
            camera_facing_reward[excellent_mask] = 0.9 * self.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[excellent_mask]
            good_mask = (camera_gate_alignment > 0.5) & (camera_gate_alignment <= 0.866)
            camera_facing_reward[good_mask] = 0.8 * self.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[good_mask]
            moderate_mask = (camera_gate_alignment > 0.0) & (camera_gate_alignment <= 0.5)
            camera_facing_reward[moderate_mask] = 0.4 * self.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[moderate_mask]
            poor_mask = (camera_gate_alignment > -0.707) & (camera_gate_alignment <= 0.0)
            camera_facing_reward[poor_mask] = 0.2 * self.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[poor_mask]
            severe_mask = camera_gate_alignment <= -0.707
            camera_facing_reward[severe_mask] = 2.0 * self.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[severe_mask]
            
            # Action penalties - FIXED: Added missing Y-action penalties for 4D action space
            action_diff = action - prev_action
            
            # ENHANCED ACTION DEBUG: Deep investigation of action tracking system
            if self.num_task_steps % 200 == 0:
                avg_action_diff = torch.mean(torch.abs(action_diff), dim=0)
                max_action_diff = torch.max(torch.abs(action_diff), dim=0)[0]
                
                # Show actual action values to understand the pattern
                avg_current = torch.mean(action, dim=0)
                avg_previous = torch.mean(prev_action, dim=0)
                
                # Check if all actions are identical across environments
                action_std = torch.std(action, dim=0)
                prev_action_std = torch.std(prev_action, dim=0)
                
                # COMMENTED OUT: Verbose action debug logs (clutters training output)
                # logger.warning(f"🔧 ACTION DEBUG - Action differences (avg): X={avg_action_diff[0]:.6f}, Y={avg_action_diff[1]:.6f}, Z={avg_action_diff[2]:.6f}, Yaw={avg_action_diff[3]:.6f}")
                # logger.warning(f"🔧 ACTION DEBUG - Action differences (max): X={max_action_diff[0]:.6f}, Y={max_action_diff[1]:.6f}, Z={max_action_diff[2]:.6f}, Yaw={max_action_diff[3]:.6f}")
                # logger.warning(f"🔧 ACTION DEBUG - Current actions (avg): X={avg_current[0]:.6f}, Y={avg_current[1]:.6f}, Z={avg_current[2]:.6f}, Yaw={avg_current[3]:.6f}")
                # logger.warning(f"🔧 ACTION DEBUG - Previous actions (avg): X={avg_previous[0]:.6f}, Y={avg_previous[1]:.6f}, Z={avg_previous[2]:.6f}, Yaw={avg_previous[3]:.6f}")
                # logger.warning(f"🔧 ACTION DEBUG - Current action std: X={action_std[0]:.6f}, Y={action_std[1]:.6f}, Z={action_std[2]:.6f}, Yaw={action_std[3]:.6f}")
                # logger.warning(f"🔧 ACTION DEBUG - Previous action std: X={prev_action_std[0]:.6f}, Y={prev_action_std[1]:.6f}, Z={prev_action_std[2]:.6f}, Yaw={prev_action_std[3]:.6f}")
                # 
                # # Check first environment for exact values
                # logger.warning(f"🔧 ACTION DEBUG - Env[0] Current: {action[0].tolist()}")
                # logger.warning(f"🔧 ACTION DEBUG - Env[0] Previous: {prev_action[0].tolist()}")
                # logger.warning(f"🔧 ACTION DEBUG - Env[0] Difference: {action_diff[0].tolist()}")
            
            x_diff_penalty = exponential_penalty_function(
                self.task_config.reward_parameters["x_action_diff_penalty_magnitude"],
                self.task_config.reward_parameters["x_action_diff_penalty_exponent"],
                action_diff[:, 0],
            )
            # FIXED: Added missing Y-action difference penalty for debugging
            y_diff_penalty = exponential_penalty_function(
                self.task_config.reward_parameters["y_action_diff_penalty_magnitude"],
                self.task_config.reward_parameters["y_action_diff_penalty_exponent"],
                action_diff[:, 1],
            )
            z_diff_penalty = exponential_penalty_function(
                self.task_config.reward_parameters["z_action_diff_penalty_magnitude"],
                self.task_config.reward_parameters["z_action_diff_penalty_exponent"],
                action_diff[:, 2],
            )
            yawrate_diff_penalty = exponential_penalty_function(
                self.task_config.reward_parameters["yawrate_action_diff_penalty_magnitude"],
                self.task_config.reward_parameters["yawrate_action_diff_penalty_exponent"],
                action_diff[:, 3],
            )
            
            # CRITICAL FIX: Add missing absolute penalties in debugging section
            x_absolute_penalty = exponential_penalty_function(
                self.task_config.reward_parameters["x_absolute_action_penalty_magnitude"],
                self.task_config.reward_parameters["x_absolute_action_penalty_exponent"],
                action[:, 0],
            )
            y_absolute_penalty = exponential_penalty_function(
                self.task_config.reward_parameters["y_absolute_action_penalty_magnitude"],
                self.task_config.reward_parameters["y_absolute_action_penalty_exponent"],
                action[:, 1],
            )
            z_absolute_penalty = exponential_penalty_function(
                self.task_config.reward_parameters["z_absolute_action_penalty_magnitude"],
                self.task_config.reward_parameters["z_absolute_action_penalty_exponent"],
                action[:, 2],
            )
            yawrate_absolute_penalty = exponential_penalty_function(
                self.task_config.reward_parameters["yawrate_absolute_action_penalty_magnitude"],
                self.task_config.reward_parameters["yawrate_absolute_action_penalty_exponent"],
                action[:, 3],
            )
            
            action_diff_penalty = x_diff_penalty + y_diff_penalty + z_diff_penalty + yawrate_diff_penalty
            absolute_action_penalty = x_absolute_penalty + y_absolute_penalty + z_absolute_penalty + yawrate_absolute_penalty
            total_action_penalty = action_diff_penalty + absolute_action_penalty
            
            # COMMENTED OUT: Verbose penalty breakdown logs (clutters training output)
            # # ABSOLUTE PENALTY DEBUG: Check individual components
            # if self.num_task_steps % 200 == 0:
            #     logger.warning(f"🔧 PENALTY BREAKDOWN - Diff penalties (avg): X={torch.mean(x_diff_penalty).item():.6f}, Y={torch.mean(y_diff_penalty).item():.6f}, Z={torch.mean(z_diff_penalty).item():.6f}, Yaw={torch.mean(yawrate_diff_penalty).item():.6f}")
            #     logger.warning(f"🔧 PENALTY BREAKDOWN - Abs penalties (avg): X={torch.mean(x_absolute_penalty).item():.6f}, Y={torch.mean(y_absolute_penalty).item():.6f}, Z={torch.mean(z_absolute_penalty).item():.6f}, Yaw={torch.mean(yawrate_absolute_penalty).item():.6f}")
            #     logger.warning(f"🔧 PENALTY BREAKDOWN - Total diff: {torch.mean(action_diff_penalty).item():.6f}, Total abs: {torch.mean(absolute_action_penalty).item():.6f}, Grand total: {torch.mean(total_action_penalty).item():.6f}")
            
            # Calculate averages for debugging
            mult_factor = 1.0 + (0.5) * self.curriculum_progress_fraction
            avg_total_reward = torch.mean(rewards).item()
            avg_pos_reward = torch.mean(mult_factor * pos_reward).item()
            avg_very_close = torch.mean(mult_factor * very_close_reward).item()
            avg_getting_closer = torch.mean(mult_factor * getting_closer_reward).item()
            avg_gate_approach = torch.mean(mult_factor * gate_approach_reward).item()
            avg_gate_alignment = torch.mean(mult_factor * gate_alignment_reward).item()
            avg_camera_facing = torch.mean(mult_factor * camera_facing_reward).item()
            avg_action_penalty = torch.mean(total_action_penalty).item()
            avg_distance = torch.mean(dist).item()
            avg_gate_distance = torch.mean(gate_distance).item()
            avg_camera_alignment = torch.mean(camera_gate_alignment).item()
            
            logger.warning("="*80)
            logger.warning(f"🔍 COMPREHENSIVE REWARD BREAKDOWN (Step {self.num_task_steps}):")
            logger.warning(f"  📊 TOTAL REWARD:           {avg_total_reward:.3f}")
            logger.warning(f"  📍 Position Reward:        {avg_pos_reward:.3f} (dist: {avg_distance:.2f}m)")
            logger.warning(f"  🎯 Very Close Reward:      {avg_very_close:.3f}")
            logger.warning(f"  ⬆️  Getting Closer:         {avg_getting_closer:.3f}")
            logger.warning(f"  🚪 Gate Approach:          {avg_gate_approach:.3f} (gate_dist: {avg_gate_distance:.2f}m)")
            logger.warning(f"  ✅ Gate Alignment:         {avg_gate_alignment:.3f}")
            logger.warning(f"  📹 Camera Facing:          {avg_camera_facing:.3f} (align: {avg_camera_alignment:.3f})")
            logger.warning(f"  🎮 Action Penalty:         {avg_action_penalty:.3f}")
            logger.warning(f"  ⚡ Multiplier Factor:      {mult_factor:.3f}")
            
            # Check for any gate passages - ADAPTIVE to gate dimensions
            curriculum_width_tolerance = self.gate_width * 0.6  # 60% of gate width
            curriculum_min_height = self.gate_position[:, 2] + self.gate_height * 0.08  # 8% above ground
            curriculum_max_height = self.gate_position[:, 2] + self.gate_height * 0.92  # 92% of gate height
            
            num_passed = torch.sum((robot_position[:, 1] > self.gate_position[:, 1]) & 
                                 (torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < curriculum_width_tolerance) &
                                 (robot_position[:, 2] > curriculum_min_height) & (robot_position[:, 2] < curriculum_max_height)).item()
            
            if num_passed > 0:
                logger.warning(f"  🎉 GATE PASSAGES:          {num_passed}/16 environments!")
                logger.warning(f"  💰 Gate Passage Reward:   {self.task_config.reward_parameters['gate_passage_reward_magnitude'].item():.1f} per passage")
            
            # Check for crashes
            num_crashes = torch.sum(obs_dict["crashes"]).item()
            if num_crashes > 0:
                logger.warning(f"  💥 CRASHES:                {num_crashes}/16 environments")
                logger.warning(f"  💸 Collision Penalty:     {self.task_config.reward_parameters['collision_penalty'].item():.1f} per crash")
            
            # EPISODE-LEVEL REWARD BREAKDOWN: Show how components contribute to episode totals
            if len(self.completed_episodes) > 0:
                logger.warning("-"*80)
                logger.warning(f"📈 EPISODE REWARD ANALYSIS (Last {len(self.completed_episodes)} Episodes):")
                
                # Calculate averages across completed episodes
                avg_episode_data = {}
                for key in self.completed_episodes[0].keys():
                    avg_episode_data[key] = sum(ep[key] for ep in self.completed_episodes) / len(self.completed_episodes)
                
                logger.warning(f"  🏆 EPISODE TOTAL:          {avg_episode_data['total_reward']:.1f}")
                logger.warning(f"  📍 Position Contribution:  {avg_episode_data['pos_reward']:.1f}")
                logger.warning(f"  🎯 Very Close Contribution: {avg_episode_data['very_close_reward']:.1f}")
                logger.warning(f"  ⬆️  Getting Closer:         {avg_episode_data['getting_closer_reward']:.1f}")
                logger.warning(f"  🚪 Gate Approach:          {avg_episode_data['gate_approach_reward']:.1f}")
                logger.warning(f"  ✅ Gate Alignment:         {avg_episode_data['gate_alignment_reward']:.1f}")
                logger.warning(f"  📹 Camera Facing:          {avg_episode_data['camera_facing_reward']:.1f}")
                logger.warning(f"  🎮 Action Penalties:       {avg_episode_data['action_penalty']:.1f}")
                logger.warning(f"  🎉 Gate Passage Bonuses:   {avg_episode_data['gate_passage_reward']:.1f} (basic + center)")
                
                # Calculate estimated passages per episode
                basic_passage_reward = 50.0  # From config
                center_bonus = 100.0  # From config
                max_reward_per_passage = (basic_passage_reward + center_bonus) * 1.5  # With curriculum multiplier
                estimated_passages = avg_episode_data['gate_passage_reward'] / max_reward_per_passage
                logger.warning(f"  📊 Estimated Passages:     {estimated_passages:.1f} per episode (should be ≤1.0)")
                logger.warning(f"  💥 Collision Penalties:    {avg_episode_data['collision_penalty']:.1f}")
                logger.warning(f"  📷 Image Penalties:        {avg_episode_data['image_reward']:.1f}")
                logger.warning(f"  📏 Average Episode Length: {avg_episode_data['episode_length']:.0f} steps")
                
                # Show recent trend (if we have enough episodes)
                if len(self.completed_episodes) >= 5:
                    recent_total = sum(ep['total_reward'] for ep in self.completed_episodes[-3:]) / 3
                    older_total = sum(ep['total_reward'] for ep in self.completed_episodes[:3]) / 3
                    trend = recent_total - older_total
                    trend_emoji = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                    logger.warning(f"  {trend_emoji} Recent Trend:         {trend:+.1f} (last 3 vs first 3)")
            
            # CURRENT EPISODE PROGRESS: Show cumulative rewards for ongoing episodes
            logger.warning("-"*80)
            logger.warning("🔄 CURRENT EPISODE PROGRESS (Cumulative):")
            
            # Average current episode progress across all environments
            avg_current_pos = torch.mean(self.episode_pos_reward).item()
            avg_current_very_close = torch.mean(self.episode_very_close_reward).item()
            avg_current_getting_closer = torch.mean(self.episode_getting_closer_reward).item()
            avg_current_gate_approach = torch.mean(self.episode_gate_approach_reward).item()
            avg_current_gate_alignment = torch.mean(self.episode_gate_alignment_reward).item()
            avg_current_camera_facing = torch.mean(self.episode_camera_facing_reward).item()
            avg_current_action_penalty = torch.mean(self.episode_action_penalty).item()
            avg_current_collision_penalty = torch.mean(self.episode_collision_penalty).item()
            avg_current_episode_length = torch.mean(self.episode_lengths).item()
            
            current_total = (avg_current_pos + avg_current_very_close + avg_current_getting_closer + 
                           avg_current_gate_approach + avg_current_gate_alignment + avg_current_camera_facing + 
                           avg_current_action_penalty + avg_current_collision_penalty)
            
            logger.warning(f"  🔄 Current Episode Total:  {current_total:.1f} (avg across 16 envs)")
            logger.warning(f"  📍 Position So Far:        {avg_current_pos:.1f}")
            logger.warning(f"  ⬆️  Getting Closer So Far:  {avg_current_getting_closer:.1f}")
            logger.warning(f"  🚪 Gate Approach So Far:   {avg_current_gate_approach:.1f}")
            logger.warning(f"  ✅ Gate Alignment So Far:  {avg_current_gate_alignment:.1f}")
            logger.warning(f"  📹 Camera Facing So Far:   {avg_current_camera_facing:.1f}")
            logger.warning(f"  💥 Collision Penalties:    {avg_current_collision_penalty:.1f}")
            logger.warning(f"  📏 Steps So Far:           {avg_current_episode_length:.0f}")
            
            logger.warning("="*80)
        
        # Store camera alignment for debugging
        self.camera_alignment_debug = camera_gate_alignment
        
        return rewards, crashes, camera_gate_alignment

    def check_and_update_curriculum_level(self, successes, crashes, timeouts):
        """
        COMPREHENSIVE MULTI-ASPECT CURRICULUM LEARNING SYSTEM
        
        Updates curriculum level and applies changes to multiple difficulty aspects:
        1. Obstacle count behind gate (increases by 1 per level, cap at 10)
        2. Drone spawning difficulty (angle and distance from gate)
        3. Drone orientation randomization (progressive random orientations)
        4. Static camera positioning (progressive angle and distance changes)
        """
        self.success_aggregate += torch.sum(successes)
        self.crashes_aggregate += torch.sum(crashes)
        self.timeouts_aggregate += torch.sum(timeouts)

        instances = self.success_aggregate + self.crashes_aggregate + self.timeouts_aggregate
        
        # Remove excessive debugging as requested by user

        if instances >= self.task_config.curriculum.check_after_log_instances:
            success_rate = self.success_aggregate / instances
            crash_rate = self.crashes_aggregate / instances
            timeout_rate = self.timeouts_aggregate / instances
            
            # ===== CURRICULUM DEBUGGING: Log curriculum evaluation =====
            old_level = self.curriculum_level
            self.log_curriculum_update(f"[CURRICULUM UPDATE] EVALUATING curriculum after {instances} instances:")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Success rate: {success_rate:.3f}")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Crash rate: {crash_rate:.3f}")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Timeout rate: {timeout_rate:.3f}")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Current level: {old_level} (max reached: {self.max_curriculum_level_reached})")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Threshold for increase: >{self.task_config.curriculum.success_rate_for_increase:.3f} (NO DECREASE POLICY)")

            # NO-DECREASE POLICY: Only allow increases, never decreases
            if success_rate > self.task_config.curriculum.success_rate_for_increase:
                self.curriculum_level += self.task_config.curriculum.increase_step
                self.max_curriculum_level_reached = max(self.max_curriculum_level_reached, self.curriculum_level)
                self.log_curriculum_update(f"[CURRICULUM UPDATE] LEVEL INCREASED: {old_level} -> {self.curriculum_level} (success rate {success_rate:.3f} > threshold)")
                self.log_curriculum_update(f"[CURRICULUM UPDATE] NEW MAX LEVEL: {self.max_curriculum_level_reached}")
            else:
                # NO-DECREASE POLICY: Only increase or stay the same, never decrease
                self.log_curriculum_update(f"[CURRICULUM UPDATE] LEVEL UNCHANGED: {self.curriculum_level} (success rate {success_rate:.3f} <= threshold, no decrease allowed)")

            # Clamp curriculum_level to valid range
            self.curriculum_level = min(
                max(self.curriculum_level, self.task_config.curriculum.min_level),
                self.task_config.curriculum.max_level,
            )
            self.obs_dict["curriculum_level"] = self.curriculum_level
            
            # Propagate curriculum level to env manager for gate unlocking
            if hasattr(self, 'sim_env') and hasattr(self.sim_env, 'global_tensor_dict'):
                self.sim_env.global_tensor_dict["curriculum_level"] = int(self.curriculum_level)
                # Re-apply gate selection so changes take effect immediately on next reset
                if hasattr(self.sim_env, 'apply_gate_variant_selection'):
                    self.sim_env.apply_gate_variant_selection(env_ids=torch.arange(self.sim_env.num_envs, device=self.device))
            
            # ===== MULTI-ASPECT CURRICULUM APPLICATION =====
            
            # 1. OBSTACLE COUNT PROGRESSION: Apply new obstacle count behind gate
            obstacles_behind_gate = self.task_config.curriculum.get_obstacle_count_behind_gate(self.curriculum_level)
            
            # FIXED CALCULATION: Account for visible assets only (not all loaded assets)
            # Even though 11 gate variants are loaded, only 1 is visible at any time
            visible_gates = 1  # Only 1 gate visible at a time (others hidden by gate selection system)
            walls = 6  # 6 boundary walls  
            robot = 0  # Robot is NOT part of env_asset_state_tensor (handled separately)
            fixed_assets_visible = visible_gates + walls  # = 7 visible fixed assets
                
            total_obstacles_in_env = fixed_assets_visible + obstacles_behind_gate
            self.obs_dict["num_obstacles_in_env"] = total_obstacles_in_env
            # logger.warning(f"[OBSTACLE_DEBUG] Curriculum update: Level {self.curriculum_level} -> fixed {fixed_assets_visible} (1 gate + 6 walls), curriculum {obstacles_behind_gate}, total {total_obstacles_in_env}")
            
            # CRITICAL: Also update the environment manager's global tensor dict for asset management
            # This ensures the asset manager gets the updated obstacle count when environments reset
            if hasattr(self.sim_env, 'global_tensor_dict'):
                old_count = self.sim_env.global_tensor_dict.get("num_obstacles_in_env", 0)
                self.sim_env.global_tensor_dict["num_obstacles_in_env"] = total_obstacles_in_env
                # logger.warning(f"[OBSTACLE_DEBUG] Curriculum update: Updated global_tensor_dict from {old_count} to {total_obstacles_in_env}")
            
            # CRITICAL FIX: Force asset manager to update obstacle count
            # The asset manager may be caching the initial obstacle count, so we need to force it to update
            if hasattr(self.sim_env, 'asset_manager'):
                try:
                    # Try to directly update the asset manager's obstacle count
                    if hasattr(self.sim_env.asset_manager, 'num_obstacles_per_env'):
                        old_count = getattr(self.sim_env.asset_manager, 'num_obstacles_per_env', 'unknown')
                        self.sim_env.asset_manager.num_obstacles_per_env = total_obstacles_in_env
                        self.log_curriculum_update(f"[CRITICAL FIX] Direct asset manager update: {old_count} → {total_obstacles_in_env}")
                        
                    # NOTE: Asset manager changes will be applied when environments naturally reset
                    self.log_curriculum_update(f"[CRITICAL FIX] Asset manager updated - changes will apply on next environment reset")
                    
                except Exception as e:
                    self.log_curriculum_update(f"[CRITICAL FIX] Warning: Failed to directly update asset manager: {e}")
            
            # ALTERNATIVE: Try to access environment configuration directly
            if hasattr(self.sim_env, 'env_config'):
                try:
                    if hasattr(self.sim_env.env_config, 'num_obstacles'):
                        old_env_count = getattr(self.sim_env.env_config, 'num_obstacles', 'unknown')
                        self.sim_env.env_config.num_obstacles = total_obstacles_in_env
                        self.log_curriculum_update(f"[CRITICAL FIX] Environment config update: {old_env_count} → {total_obstacles_in_env}")
                except Exception as e:
                    self.log_curriculum_update(f"[CRITICAL FIX] Warning: Failed to update environment config: {e}")
            
            # 2. STATIC CAMERA DIFFICULTY: Update camera parameters for NEW episodes only
            # Update max camera angle for logging (affects new episodes only)
            self.max_camera_angle, self.camera_height_offset, self.camera_distance_offset = self.task_config.curriculum.get_static_camera_difficulty(self.curriculum_level)
            
            # DON'T update camera positions here - only update on episode reset
            # This ensures camera orientation stays fixed during each episode
            self.log_curriculum_update(f"[CAMERA UPDATE] Camera max angle updated for NEW episodes: ±{self.max_camera_angle:.1f}° (existing episodes unchanged)")

            # Calculate curriculum progress fraction
            self.curriculum_progress_fraction = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

            # ===== COMPREHENSIVE CURRICULUM LOGGING =====
            self.log_curriculum_update(f"Gate Navigation Curriculum Level: {self.curriculum_level}, Progress: {self.curriculum_progress_fraction:.3f}")
            self.log_curriculum_update(f"\nSuccess Rate: {success_rate:.3f}\nCrash Rate: {crash_rate:.3f}\nTimeout Rate: {timeout_rate:.3f}")
            
            self.log_curriculum_update(f"\nCURRICULUM APPLIED:")
            self.log_curriculum_update(f"   1. OBSTACLES: {obstacles_behind_gate} behind gate (total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)")
            try:
                sr = self.task_config.curriculum.get_spawn_ranges(self.curriculum_level)
                self.log_curriculum_update(
                    f"   2. SPAWN: X∈[{(-sr['x_half_span_m']):.1f}, {(+sr['x_half_span_m']):.1f}] m, "
                    f"Y∈[{(sr['y_center_m']-sr['y_half_span_m']):.1f}, {(sr['y_center_m']+sr['y_half_span_m']):.1f}] m, "
                    f"Z∈[{(sr['z_center_m']-sr['z_half_span_m']):.1f}, {(sr['z_center_m']+sr['z_half_span_m']):.1f}] m; yaw ±{(sr['yaw_abs_rad']*57.2958):.1f}°"
                )
            except Exception as e:
                self.log_curriculum_update(f"   2. SPAWN: (fallback) Using fixed LMF2 config due to: {e}")
            # Get current randomized angle for first environment (representative)
            current_angle = 0.0
            if hasattr(self, 'static_camera_manager') and hasattr(self.static_camera_manager, 'current_camera_angles'):
                current_angle = self.static_camera_manager.current_camera_angles[0] if self.static_camera_manager.current_camera_angles else 0.0
            self.log_curriculum_update(f"   3. CAMERA ANGLE: ±{self.max_camera_angle:.1f}deg max range, env0: {current_angle:.1f}deg (fixed per episode)")
            
            # 4. GATE SIZE UNLOCKS (Curriculum-gated randomization)
            if hasattr(self.sim_env, 'global_tensor_dict'):
                gate_names = []
                if len(self.sim_env.global_tensor_dict.get("gate_variant_names_per_env", [])) > 0:
                    gate_names = self.sim_env.global_tensor_dict["gate_variant_names_per_env"][0]
                # Compute linear threshold from 80 -> 40 over levels 3..23
                if self.curriculum_level <= 3:
                    min_scale = 80
                elif self.curriculum_level >= 23:
                    min_scale = 40
                else:
                    frac = (self.curriculum_level - 3) / (23 - 3)
                    raw = 80 - frac * (80 - 40)
                    min_scale = int((int(raw) // 2) * 2)
                    if min_scale < 40:
                        min_scale = 40
                    if min_scale > 100:
                        min_scale = 100
                # Collect scales meeting threshold
                scales = []
                for n in gate_names:
                    if isinstance(n, str) and "gate_scale_" in n:
                        try:
                            s = int(n.replace("gate_scale_", ""))
                            if s >= min_scale:
                                scales.append(s)
                        except:
                            pass
                # Report unique scales only (avoid duplicates from config classes)
                scales = sorted(list(set(scales)), reverse=True)
                self.log_curriculum_update(f"   4. GATE SIZE: unlocked scales >= {min_scale}% -> {scales if scales else [100]} (uniform across unique scales)")
            
            # 5. CAMERA NOISE PROGRESSION (D455 Simulation)
            camera_gaussian_std, camera_dropout_rate = self.task_config.curriculum.get_camera_noise(self.curriculum_level)
            self.log_curriculum_update(f"   5. CAMERA NOISE: Gaussian STD={camera_gaussian_std:.4f}, Dropout={camera_dropout_rate*100:.1f}% (both drone & static)")
            
            # 6. CAMERA FRAME DROPOUT (entire-frame)
            fd = self.task_config.curriculum.get_camera_frame_dropout(self.curriculum_level)
            self.log_curriculum_update(
                f"   6. CAMERA FRAME DROPOUT: drone_total={fd['drone_total']*100:.1f}% (freeze {fd['drone_freeze']*100:.1f}%, blank {fd['drone_blank']*100:.1f}%), "
                f"static_total={fd['static_total']*100:.1f}% (freeze {fd['static_freeze']*100:.1f}%, blank {fd['static_blank']*100:.1f}%)"
            )
            
            # 7. STATE NOISE (pose)
            if getattr(self.task_config.curriculum, "enable_state_noise", False):
                sn = self.task_config.curriculum.get_state_noise(self.curriculum_level)
                self.log_curriculum_update(
                    f"   7. STATE NOISE: drone_pos_std={sn['drone_pos_std_m']:.4f} m, drone_orient_std={sn['drone_orient_std_rad']*57.2958:.3f} deg, "
                    f"static_pos_std={sn['static_pos_std_m']:.4f} m, static_orient_std={sn['static_orient_std_rad']*57.2958:.3f} deg"
                )
            else:
                self.log_curriculum_update("   7. STATE NOISE: disabled")
            
            # ===== CURRICULUM DEBUGGING: Final state after update =====
            self.log_curriculum_update(f"[CURRICULUM UPDATE] FINAL STATE:")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Level: {self.curriculum_level} (range: {self.task_config.curriculum.min_level}-{self.task_config.curriculum.max_level})")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Max level reached: {self.max_curriculum_level_reached} (NO-DECREASE POLICY)")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Progress: {self.curriculum_progress_fraction:.3f}")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Obstacles behind gate: {obstacles_behind_gate} (total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Asset manager: Updated both obs_dict and global_tensor_dict with count {total_obstacles_in_env}")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Spawn difficulty: LMF2 config with ±0.5m lateral, ±45° orientation (no curriculum dependency)")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Camera angle: ±{self.max_camera_angle:.1f}deg max range (randomized per episode reset, fixed during episode)")
            
            # ===== END CURRICULUM DEBUGGING =====
            
            # Add comprehensive curriculum metrics to infos for wandb logging
            self.infos["curriculum/level"] = torch.tensor(self.curriculum_level, dtype=torch.float32)
            self.infos["curriculum/progress"] = torch.tensor(self.curriculum_progress_fraction, dtype=torch.float32)
            self.infos["curriculum/success_rate"] = torch.tensor(success_rate, dtype=torch.float32)
            self.infos["curriculum/crash_rate"] = torch.tensor(crash_rate, dtype=torch.float32)
            self.infos["curriculum/timeout_rate"] = torch.tensor(timeout_rate, dtype=torch.float32)
            
            # Add curriculum metrics
            self.infos["curriculum/obstacles_behind_gate"] = torch.tensor(obstacles_behind_gate, dtype=torch.float32)
            self.infos["curriculum/total_assets"] = torch.tensor(total_obstacles_in_env, dtype=torch.float32)
            self.infos["curriculum/max_level_reached"] = torch.tensor(self.max_curriculum_level_reached, dtype=torch.float32)
            
            # Add camera noise metrics (D455 simulation)
            self.infos["curriculum/camera_gaussian_std"] = torch.tensor(camera_gaussian_std, dtype=torch.float32)
            self.infos["curriculum/camera_dropout_rate"] = torch.tensor(camera_dropout_rate, dtype=torch.float32)
            # Add camera frame dropout metrics
            fd = self.task_config.curriculum.get_camera_frame_dropout(self.curriculum_level)
            self.infos["curriculum/camera_frame_dropout_drone_total"] = torch.tensor(fd["drone_total"], dtype=torch.float32)
            self.infos["curriculum/camera_frame_dropout_static_total"] = torch.tensor(fd["static_total"], dtype=torch.float32)
            self.infos["curriculum/camera_frame_freeze_drone"] = torch.tensor(fd["drone_freeze"], dtype=torch.float32)
            self.infos["curriculum/camera_frame_blank_drone"] = torch.tensor(fd["drone_blank"], dtype=torch.float32)
            self.infos["curriculum/camera_frame_freeze_static"] = torch.tensor(fd["static_freeze"], dtype=torch.float32)
            self.infos["curriculum/camera_frame_blank_static"] = torch.tensor(fd["static_blank"], dtype=torch.float32)
            
            # Add camera angle metrics
            self.infos["curriculum/camera_max_angle"] = torch.tensor(self.max_camera_angle, dtype=torch.float32)
            # Use first environment's angle as representative for wandb tracking
            current_angle = 0.0
            if hasattr(self, 'static_camera_manager') and hasattr(self.static_camera_manager, 'current_camera_angles'):
                current_angle = self.static_camera_manager.current_camera_angles[0] if self.static_camera_manager.current_camera_angles else 0.0
            self.infos["curriculum/camera_current_angle"] = torch.tensor(current_angle, dtype=torch.float32)
            
            # Add state noise metrics
            if getattr(self.task_config.curriculum, "enable_state_noise", False):
                sn = self.task_config.curriculum.get_state_noise(self.curriculum_level)
                self.infos["curriculum/state_noise_drone_pos_std_m"] = torch.tensor(sn["drone_pos_std_m"], dtype=torch.float32)
                self.infos["curriculum/state_noise_drone_orient_std_deg"] = torch.tensor(sn["drone_orient_std_rad"]*57.2958, dtype=torch.float32)
                self.infos["curriculum/state_noise_static_pos_std_m"] = torch.tensor(sn["static_pos_std_m"], dtype=torch.float32)
                self.infos["curriculum/state_noise_static_orient_std_deg"] = torch.tensor(sn["static_orient_std_rad"]*57.2958, dtype=torch.float32)
            
            self.log_curriculum_update(f"[CURRICULUM UPDATE] RESETTING counters for next evaluation period")
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

    def update_episode_reward_tracking(self, obs_dict, rewards, crashes):
        """Update cumulative episode reward tracking for comprehensive debugging."""
        robot_position = obs_dict["robot_position"]
        
        # Calculate individual reward components (same as in compute_rewards_and_crashes)
        dist = torch.norm(self.pos_error_vehicle_frame, dim=1)
        prev_dist = torch.norm(self.pos_error_vehicle_frame_prev, dim=1)
        # CRITICAL FIX: Clone action tensors here too for consistency
        action = obs_dict["robot_actions"].clone()
        prev_action = obs_dict["robot_prev_actions"].clone()
        
        mult_factor = 1.0 + (0.5) * self.curriculum_progress_fraction
        
        # Position reward
        pos_reward = exponential_reward_function(
            self.task_config.reward_parameters["pos_reward_magnitude"],
            self.task_config.reward_parameters["pos_reward_exponent"],
            dist,
        )
        self.episode_pos_reward += mult_factor * pos_reward
        
        # Very close reward
        very_close_reward = exponential_reward_function(
            self.task_config.reward_parameters["very_close_to_goal_reward_magnitude"],
            self.task_config.reward_parameters["very_close_to_goal_reward_exponent"],
            dist,
        )
        self.episode_very_close_reward += mult_factor * very_close_reward
        
        # Getting closer reward
        getting_closer = prev_dist - dist
        getting_closer_reward = torch.where(
            getting_closer > 0,
            self.task_config.reward_parameters["getting_closer_reward_multiplier"] * getting_closer,
            2.0 * self.task_config.reward_parameters["getting_closer_reward_multiplier"] * getting_closer,
        )
        self.episode_getting_closer_reward += mult_factor * getting_closer_reward
        
        # Gate approach reward
        gate_distance = torch.norm(robot_position - self.gate_position, dim=1)
        gate_approach_reward = exponential_reward_function(
            self.task_config.reward_parameters["gate_approach_reward_magnitude"],
            0.5,
            gate_distance,
        )
        self.episode_gate_approach_reward += mult_factor * gate_approach_reward
        
        # Gate alignment reward
        gate_alignment_reward = torch.zeros_like(gate_distance)
        aligned_mask = torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < 1.5
        gate_alignment_reward[aligned_mask] = self.task_config.reward_parameters["gate_alignment_reward_magnitude"]
        self.episode_gate_alignment_reward += mult_factor * gate_alignment_reward
        
        # Camera facing reward (same calculation as in debugging section)
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        drone_to_gate = self.gate_position - robot_position
        drone_to_gate_normalized = drone_to_gate / (torch.norm(drone_to_gate, dim=1, keepdim=True) + 1e-8)
        
        # Get drone's forward direction (where camera points)
        qw, qx, qy, qz = robot_vehicle_orientation[:, 3], robot_vehicle_orientation[:, 0], robot_vehicle_orientation[:, 1], robot_vehicle_orientation[:, 2]
        forward_x = 1.0 - 2.0 * (qy * qy + qz * qz)
        forward_y = 2.0 * (qx * qy + qw * qz)
        forward_z = 2.0 * (qx * qz - qw * qy)
        drone_forward = torch.stack([forward_x, forward_y, forward_z], dim=1)
        drone_forward_normalized = drone_forward / (torch.norm(drone_forward, dim=1, keepdim=True) + 1e-8)
        
        # Calculate alignment and camera facing reward
        camera_gate_alignment = torch.sum(drone_forward_normalized * drone_to_gate_normalized, dim=1)
        camera_gate_alignment = torch.clamp(camera_gate_alignment, -1.0, 1.0)
        
        camera_facing_reward = torch.zeros_like(camera_gate_alignment)
        perfect_mask = camera_gate_alignment > 0.966
        camera_facing_reward[perfect_mask] = self.task_config.reward_parameters["camera_facing_reward_magnitude"]
        excellent_mask = (camera_gate_alignment > 0.866) & (camera_gate_alignment <= 0.966)
        camera_facing_reward[excellent_mask] = 0.9 * self.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[excellent_mask]
        good_mask = (camera_gate_alignment > 0.5) & (camera_gate_alignment <= 0.866)
        camera_facing_reward[good_mask] = 0.8 * self.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[good_mask]
        moderate_mask = (camera_gate_alignment > 0.0) & (camera_gate_alignment <= 0.5)
        camera_facing_reward[moderate_mask] = 0.4 * self.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[moderate_mask]
        poor_mask = (camera_gate_alignment > -0.707) & (camera_gate_alignment <= 0.0)
        camera_facing_reward[poor_mask] = 0.2 * self.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[poor_mask]
        severe_mask = camera_gate_alignment <= -0.707
        camera_facing_reward[severe_mask] = 2.0 * self.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[severe_mask]
        self.episode_camera_facing_reward += mult_factor * camera_facing_reward
        
        # Action penalties - FIXED: Added missing Y-action penalties for 4D action space  
        action_diff = action - prev_action
        
        x_diff_penalty = exponential_penalty_function(
            self.task_config.reward_parameters["x_action_diff_penalty_magnitude"],
            self.task_config.reward_parameters["x_action_diff_penalty_exponent"],
            action_diff[:, 0],
        )
        # FIXED: Added missing Y-action difference penalty for episode tracking
        y_diff_penalty = exponential_penalty_function(
            self.task_config.reward_parameters["y_action_diff_penalty_magnitude"],
            self.task_config.reward_parameters["y_action_diff_penalty_exponent"],
            action_diff[:, 1],
        )
        z_diff_penalty = exponential_penalty_function(
            self.task_config.reward_parameters["z_action_diff_penalty_magnitude"],
            self.task_config.reward_parameters["z_action_diff_penalty_exponent"],
            action_diff[:, 2],
        )
        yawrate_diff_penalty = exponential_penalty_function(
            self.task_config.reward_parameters["yawrate_action_diff_penalty_magnitude"],
            self.task_config.reward_parameters["yawrate_action_diff_penalty_exponent"],
            action_diff[:, 3],
        )
        
        # CRITICAL FIX: Add missing absolute penalties in episode tracking
        x_absolute_penalty = exponential_penalty_function(
            self.task_config.reward_parameters["x_absolute_action_penalty_magnitude"],
            self.task_config.reward_parameters["x_absolute_action_penalty_exponent"],
            action[:, 0],
        )
        y_absolute_penalty = exponential_penalty_function(
            self.task_config.reward_parameters["y_absolute_action_penalty_magnitude"],
            self.task_config.reward_parameters["y_absolute_action_penalty_exponent"],
            action[:, 1],
        )
        z_absolute_penalty = exponential_penalty_function(
            self.task_config.reward_parameters["z_absolute_action_penalty_magnitude"],
            self.task_config.reward_parameters["z_absolute_action_penalty_exponent"],
            action[:, 2],
        )
        yawrate_absolute_penalty = exponential_penalty_function(
            self.task_config.reward_parameters["yawrate_absolute_action_penalty_magnitude"],
            self.task_config.reward_parameters["yawrate_absolute_action_penalty_exponent"],
            action[:, 3],
        )
        
        action_diff_penalty = x_diff_penalty + y_diff_penalty + z_diff_penalty + yawrate_diff_penalty
        absolute_action_penalty = x_absolute_penalty + y_absolute_penalty + z_absolute_penalty + yawrate_absolute_penalty
        total_action_penalty = action_diff_penalty + absolute_action_penalty
        self.episode_action_penalty += total_action_penalty
        
        # Track collision penalties
        collision_mask = crashes > 0
        collision_penalty = torch.where(
            collision_mask,
            self.task_config.reward_parameters["collision_penalty"],
            torch.zeros_like(crashes, dtype=torch.float32),
        )
        self.episode_collision_penalty += collision_penalty
        
        # Track gate passage rewards (check if any gate passages occurred this step) - ADAPTIVE
        # Use the same logic as main reward system with adaptive dimensions
        tracking_width_tolerance = self.gate_width * 0.6  # 60% of gate width
        tracking_min_height = self.gate_position[:, 2] + self.gate_height * 0.08  # 8% above ground
        tracking_max_height = self.gate_position[:, 2] + self.gate_height * 0.92  # 92% of gate height
        
        gate_passed_this_step = (
            (robot_position[:, 1] > self.gate_position[:, 1]) &
            (torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < tracking_width_tolerance) &
            (robot_position[:, 2] > tracking_min_height) & (robot_position[:, 2] < tracking_max_height) &
            (~self.gate_passed)  # Haven't passed before
        )
        
        # Center passage detection with adaptive dimensions (like main system)
        x_distance_from_center = torch.abs(robot_position[:, 0] - self.gate_position[:, 0])
        z_distance_from_center = torch.abs(robot_position[:, 2] - (self.gate_position[:, 2] + self.gate_center_height))
        
        # Adaptive center thresholds
        center_x_threshold = self.gate_width * 0.2  # 20% of gate width for center alignment
        center_z_threshold = self.gate_height * 0.125  # 12.5% of gate height for center alignment
        center_aligned_mask = (x_distance_from_center < center_x_threshold) & (z_distance_from_center < center_z_threshold)
        
        # Basic gate passage reward
        gate_passage_reward = torch.where(
            gate_passed_this_step,
            mult_factor * self.task_config.reward_parameters["gate_passage_reward_magnitude"],
            torch.zeros_like(gate_passed_this_step, dtype=torch.float32),
        )
        
        # Gate center passage bonus (only for centered passages)
        gate_center_passage_bonus = torch.where(
            gate_passed_this_step & center_aligned_mask,
            mult_factor * self.task_config.reward_parameters["gate_center_passage_bonus_magnitude"],
            torch.zeros_like(gate_passed_this_step, dtype=torch.float32),
        )
        
        # CRITICAL FIX: Update gate_passed flag to prevent multiple detections in same episode
        self.gate_passed = self.gate_passed | gate_passed_this_step
        
        # Track total gate passage rewards (basic + center bonus)
        total_gate_rewards = gate_passage_reward + gate_center_passage_bonus
        self.episode_gate_passage_reward += total_gate_rewards
        
        # Track image rewards (from post_image_reward_addition)
        if hasattr(self, 'min_pixel_dist'):
            image_rewards = -exponential_reward_function(
                4.0, 1.0, self.min_pixel_dist[self.terminations < 0]
            )
            # Only add for non-terminated environments
            non_terminated_mask = self.terminations < 0
            if torch.sum(non_terminated_mask) > 0:
                self.episode_image_reward[non_terminated_mask] += image_rewards
        
        # Increment episode length tracking
        self.episode_lengths += 1

    def reset_episode_reward_tracking(self, env_ids):
        """Reset episode reward tracking for specified environments when episodes end."""
        if len(env_ids) == 0:
            return
            
        # Store completed episode data for averaging
        for env_id in env_ids:
            if self.episode_lengths[env_id] > 0:  # Valid episode
                episode_data = {
                    'total_reward': (
                        self.episode_pos_reward[env_id] + 
                        self.episode_very_close_reward[env_id] + 
                        self.episode_getting_closer_reward[env_id] + 
                        self.episode_gate_approach_reward[env_id] + 
                        self.episode_gate_alignment_reward[env_id] + 
                        self.episode_camera_facing_reward[env_id] + 
                        self.episode_action_penalty[env_id] + 
                        self.episode_gate_passage_reward[env_id] + 
                        self.episode_collision_penalty[env_id] + 
                        self.episode_image_reward[env_id]
                    ).item(),
                    'pos_reward': self.episode_pos_reward[env_id].item(),
                    'very_close_reward': self.episode_very_close_reward[env_id].item(),
                    'getting_closer_reward': self.episode_getting_closer_reward[env_id].item(),
                    'gate_approach_reward': self.episode_gate_approach_reward[env_id].item(),
                    'gate_alignment_reward': self.episode_gate_alignment_reward[env_id].item(),
                    'camera_facing_reward': self.episode_camera_facing_reward[env_id].item(),
                    'action_penalty': self.episode_action_penalty[env_id].item(),
                    'gate_passage_reward': self.episode_gate_passage_reward[env_id].item(),  # Now includes both basic + center bonus
                    'collision_penalty': self.episode_collision_penalty[env_id].item(),
                    'image_reward': self.episode_image_reward[env_id].item(),
                    'episode_length': self.episode_lengths[env_id].item(),
                }
                self.completed_episodes.append(episode_data)
                
                # Keep only last N episodes
                if len(self.completed_episodes) > self.max_stored_episodes:
                    self.completed_episodes.pop(0)
        
        # Reset trackers for completed episodes
        self.episode_pos_reward[env_ids] = 0
        self.episode_very_close_reward[env_ids] = 0
        self.episode_getting_closer_reward[env_ids] = 0
        self.episode_gate_approach_reward[env_ids] = 0
        self.episode_gate_alignment_reward[env_ids] = 0
        self.episode_camera_facing_reward[env_ids] = 0
        self.episode_action_penalty[env_ids] = 0
        self.episode_gate_passage_reward[env_ids] = 0
        self.episode_collision_penalty[env_ids] = 0
        self.episode_image_reward[env_ids] = 0
        self.episode_lengths[env_ids] = 0


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
        
        # Per-environment camera angle tracking - FIXED during each episode
        self.num_envs = len(self.env_handles)
        self.current_camera_angles = [0.0] * self.num_envs  # Track angle per environment
        
        self._setup_static_camera()
    
    def get_average_camera_angle(self):
        """Get average camera angle across all environments for logging."""
        if not hasattr(self, 'current_camera_angles') or not self.current_camera_angles:
            return 0.0
        return sum(self.current_camera_angles) / len(self.current_camera_angles)
    
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
        """Update static camera orientation ONLY for resetting environments."""
        if hasattr(self, 'use_synthetic_camera') and self.use_synthetic_camera:
            # In synthetic mode, just update the angle tracking for the resetting environments
            from aerial_gym.config.task_config.navigation_task_config_gate import task_config
            max_angle_range, _, _ = task_config.curriculum.get_static_camera_difficulty(curriculum_level)
            
            import random
            for env_idx in env_ids:
                if env_idx < len(self.current_camera_angles):
                    if max_angle_range > 0:
                        self.current_camera_angles[env_idx] = random.uniform(-max_angle_range, max_angle_range)
                    else:
                        self.current_camera_angles[env_idx] = 0.0
            
            logger.debug(f"Synthetic camera mode - updated angles for envs {env_ids.tolist()}")
            return
            
        if not self.camera_setup_success or len(self.camera_handles) == 0:
            return
        
        # Get maximum angle range from curriculum (linear progression from 0° to ±30°)
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        max_angle_range, _, _ = task_config.curriculum.get_static_camera_difficulty(curriculum_level)
        
        try:
            # Fixed camera position (3m behind gate, 1.5m height) - POSITION NEVER CHANGES
            base_camera_pos = gymapi.Vec3(0.0, -3.0, 1.5)
            
            import math
            import random
            
            # Update camera orientation ONLY for the specified environments (those resetting)
            for env_idx in env_ids:
                if env_idx >= len(self.env_handles) or env_idx >= len(self.camera_handles):
                    continue
                    
                # Generate NEW random angle for this resetting environment
                if max_angle_range > 0:
                    angle_offset_degrees = random.uniform(-max_angle_range, max_angle_range)
                else:
                    angle_offset_degrees = 0.0
                
                # Store the angle for this environment
                if env_idx < len(self.current_camera_angles):
                    self.current_camera_angles[env_idx] = angle_offset_degrees
                
                # Convert to radians and update camera
                angle_offset_radians = angle_offset_degrees * (3.14159 / 180.0)
                
                # Calculate offset target position based on randomized angle for this environment
                target_distance = 3.0  # Distance from camera to gate
                target_x = base_camera_pos.x + target_distance * math.sin(angle_offset_radians)
                target_y = base_camera_pos.y + target_distance * math.cos(angle_offset_radians)
                target_z = 1.5  # Keep same height as gate center
                
                new_target = gymapi.Vec3(target_x, target_y, target_z)
                
                # Update ONLY this environment's camera
                env_handle = self.env_handles[env_idx]
                cam_handle = self.camera_handles[env_idx]
                self.gym.set_camera_location(cam_handle, env_handle, base_camera_pos, new_target)
                
                logger.debug(f"Updated static camera for env {env_idx} - Level {curriculum_level}: {angle_offset_degrees:.1f}° (max range: ±{max_angle_range:.1f}°)")
            
            logger.debug(f"Updated static camera orientation for {len(env_ids)} resetting environments")
            
        except Exception as e:
            logger.warning(f"Failed to update static camera orientation: {e}")
            # Fall back to fixed positioning if update fails
            logger.debug(f"Static camera orientation update failed - using fixed positioning")
            return
    
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
            
            # Get images from camera 0 (any env, all envs share same viewpoint)
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
    gate_width,
    gate_height,
    gate_center_height,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor], Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
    
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
    
    # Action penalties - FIXED: Added missing Y-action penalties for 4D action space
    action_diff = action - prev_action
    x_diff_penalty = exponential_penalty_function(
        parameter_dict["x_action_diff_penalty_magnitude"],
        parameter_dict["x_action_diff_penalty_exponent"],
        action_diff[:, 0],
    )
    # FIXED: Added missing Y-action difference penalty for 4D action space [x_vel, y_vel, z_vel, yaw_rate]
    y_diff_penalty = exponential_penalty_function(
        parameter_dict["y_action_diff_penalty_magnitude"],
        parameter_dict["y_action_diff_penalty_exponent"],
        action_diff[:, 1],
    )
    z_diff_penalty = exponential_penalty_function(
        parameter_dict["z_action_diff_penalty_magnitude"],
        parameter_dict["z_action_diff_penalty_exponent"],
        action_diff[:, 2],
    )
    yawrate_diff_penalty = exponential_penalty_function(
        parameter_dict["yawrate_action_diff_penalty_magnitude"],
        parameter_dict["yawrate_action_diff_penalty_exponent"],
        action_diff[:, 3],
    )
    
    action_diff_penalty = x_diff_penalty + y_diff_penalty + z_diff_penalty + yawrate_diff_penalty
    
    # Absolute action penalties - FIXED: Removed curriculum scaling and added Y-axis penalty
    x_absolute_penalty = exponential_penalty_function(
        parameter_dict["x_absolute_action_penalty_magnitude"],
        parameter_dict["x_absolute_action_penalty_exponent"],
        action[:, 0],
    )
    # FIXED: Added missing Y-action absolute penalty for 4D action space
    y_absolute_penalty = exponential_penalty_function(
        parameter_dict["y_absolute_action_penalty_magnitude"],
        parameter_dict["y_absolute_action_penalty_exponent"],
        action[:, 1],
    )
    z_absolute_penalty = exponential_penalty_function(
        parameter_dict["z_absolute_action_penalty_magnitude"],
        parameter_dict["z_absolute_action_penalty_exponent"],
        action[:, 2],
    )
    yawrate_absolute_penalty = exponential_penalty_function(
        parameter_dict["yawrate_absolute_action_penalty_magnitude"],
        parameter_dict["yawrate_absolute_action_penalty_exponent"],
        action[:, 3],
    )
    
    absolute_action_penalty = x_absolute_penalty + y_absolute_penalty + z_absolute_penalty + yawrate_absolute_penalty
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
    # Check if robot is roughly aligned with gate opening (Y direction) - ADAPTIVE to gate width
    gate_width_tolerance = gate_width * 0.6  # 60% of gate width for alignment tolerance
    aligned_mask = torch.abs(robot_position[:, 0] - gate_position[:, 0]) < gate_width_tolerance
    gate_alignment_reward[aligned_mask] = parameter_dict["gate_alignment_reward_magnitude"]
    
    # Enhanced center alignment rewards for precise gate navigation - ADAPTIVE to gate size
    gate_center_bonus = torch.zeros_like(gate_distance)
    # Distance from gate center in X direction (horizontal alignment)
    x_distance_from_center = torch.abs(robot_position[:, 0] - gate_position[:, 0])
    # Distance from gate center in Z direction (vertical alignment) - ADAPTIVE to gate center height
    z_distance_from_center = torch.abs(robot_position[:, 2] - (gate_position[:, 2] + gate_center_height))
    
    # Check if robot is very close to gate center - ADAPTIVE thresholds
    x_threshold = gate_width * 0.2  # 20% of gate width for precise X alignment
    z_threshold = gate_height * 0.125  # 12.5% of gate height for precise Z alignment
    center_aligned_mask = (x_distance_from_center < x_threshold) & (z_distance_from_center < z_threshold)
    gate_center_bonus[center_aligned_mask] = parameter_dict["gate_center_bonus_magnitude"]
    
    # Check for gate passage (crossing Y = 0 plane with proper alignment) - ADAPTIVE to gate dimensions
    gate_passage_width_tolerance = gate_width * 0.6  # 60% of gate width for passage detection
    gate_min_height = gate_position[:, 2] + gate_height * 0.1  # 10% above ground
    gate_max_height = gate_position[:, 2] + gate_height * 0.9  # 90% of gate height
    
    just_passed_gate = (
        (robot_position[:, 1] > gate_position[:, 1]) &  # In front of gate
        (torch.abs(robot_position[:, 0] - gate_position[:, 0]) < gate_passage_width_tolerance) &  # Within gate width
        (robot_position[:, 2] > gate_min_height) & (robot_position[:, 2] < gate_max_height) &  # Within gate height
        (~gate_passed)  # Haven't passed before
    )
    
    # Check for center passage (more precise alignment) - ADAPTIVE thresholds
    center_passage_x_tolerance = gate_width * 0.2  # 20% of gate width for center passage
    center_passage_z_tolerance = gate_height * 0.125  # 12.5% of gate height for center passage
    
    just_passed_center = (
        just_passed_gate &  # Basic passage requirement
        (x_distance_from_center < center_passage_x_tolerance) &  # Centered horizontally
        (z_distance_from_center < center_passage_z_tolerance)    # Centered vertically
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

    # Calculate individual component contributions (for debugging)
    multiplied_pos_reward = MULTIPLICATION_FACTOR_REWARD * pos_reward
    multiplied_very_close_reward = MULTIPLICATION_FACTOR_REWARD * very_close_to_goal_reward  
    multiplied_getting_closer = MULTIPLICATION_FACTOR_REWARD * getting_closer_reward
    multiplied_distance_reward = MULTIPLICATION_FACTOR_REWARD * distance_from_goal_reward
    multiplied_gate_approach = MULTIPLICATION_FACTOR_REWARD * gate_approach_reward
    multiplied_gate_alignment = MULTIPLICATION_FACTOR_REWARD * gate_alignment_reward
    multiplied_gate_passage = MULTIPLICATION_FACTOR_REWARD * gate_passage_reward
    multiplied_gate_center_bonus = MULTIPLICATION_FACTOR_REWARD * gate_center_bonus
    multiplied_gate_center_passage = MULTIPLICATION_FACTOR_REWARD * gate_center_passage_bonus
    multiplied_camera_facing = MULTIPLICATION_FACTOR_REWARD * camera_facing_reward
    multiplied_altitude_maintenance = MULTIPLICATION_FACTOR_REWARD * altitude_maintenance_reward

    # Combined reward - NOW INCLUDING CAMERA FACING REWARD AND ALTITUDE MAINTENANCE
    reward = (
        multiplied_pos_reward
        + multiplied_very_close_reward
        + multiplied_getting_closer
        + multiplied_distance_reward
        + multiplied_gate_approach
        + multiplied_gate_alignment
        + multiplied_gate_passage
        + multiplied_gate_center_bonus
        + multiplied_gate_center_passage
        + multiplied_camera_facing  # Camera facing reward
        + multiplied_altitude_maintenance  # NEW: Altitude maintenance reward
        + total_action_penalty
    )

    # Apply collision penalties
    reward[:] = torch.where(
        crashes > 0,
        parameter_dict["collision_penalty"] * torch.ones_like(reward),
        reward,
    )
    
    return reward, crashes, camera_gate_alignment
