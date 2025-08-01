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
        
        # Estimate fixed assets: gate (1) + walls (6) + robot (1) = 8 base assets
        estimated_fixed_assets = 8
        total_obstacles_in_env = estimated_fixed_assets + obstacles_behind_gate
        
        logger.info(f"PRE-INIT: Setting curriculum level {self.curriculum_level} with {obstacles_behind_gate} curriculum obstacles")
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

        # Gate-specific tracking
        self.gate_position = torch.zeros((self.sim_env.num_envs, 3), device=self.device)
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
        
        # Get the actual number of assets that are always kept from the asset manager
        if hasattr(self.sim_env, 'asset_manager') and hasattr(self.sim_env.asset_manager, 'num_keep_in_env'):
            fixed_assets_always_kept = self.sim_env.asset_manager.num_keep_in_env
            logger.info(f"ACTUAL: Asset manager reports {fixed_assets_always_kept} fixed assets")
        else:
            # Use the estimated value from pre-initialization
            fixed_assets_always_kept = 8  # Gate + 6 walls + robot (estimated)
            logger.warning(f"FALLBACK: Using estimated {fixed_assets_always_kept} fixed assets")
        
        total_obstacles_in_env = fixed_assets_always_kept + obstacles_behind_gate
        
        # Update observation dictionary with obstacle count
        self.obs_dict["num_obstacles_in_env"] = total_obstacles_in_env
        
        # Confirm the environment manager has the correct count
        if hasattr(self.sim_env, 'global_tensor_dict'):
            if self.sim_env.global_tensor_dict.get("num_obstacles_in_env", 0) != total_obstacles_in_env:
                logger.warning(f"MISMATCH: Updating global_tensor_dict from {self.sim_env.global_tensor_dict.get('num_obstacles_in_env', 0)} to {total_obstacles_in_env}")
                self.sim_env.global_tensor_dict["num_obstacles_in_env"] = total_obstacles_in_env
            else:
                logger.info(f"CONFIRMED: Global tensor dict already has correct obstacle count: {total_obstacles_in_env}")
        
        logger.info(f"FINAL: Fixed assets: {fixed_assets_always_kept}, Curriculum obstacles: {obstacles_behind_gate}, Total: {total_obstacles_in_env}")
        
        # Initialize camera difficulty parameters (only static camera curriculum remains)
        self.max_camera_angle, self.camera_height_offset, self.camera_distance_offset = self.task_config.curriculum.get_static_camera_difficulty(self.curriculum_level)
        
        # ===== CURRICULUM LOGGING =====
        logger.info(f"INITIAL CURRICULUM (Level {self.curriculum_level}):")
        logger.info(f"   1. OBSTACLES: {obstacles_behind_gate} behind gate (total assets: {total_obstacles_in_env} = {fixed_assets_always_kept} fixed + {obstacles_behind_gate} curriculum)")
        logger.info(f"   2. SPAWN: Using LMF2 config with ±0.5m in all directions, ±45° orientation (no curriculum dependency)")
        logger.info(f"   3. CAMERA ANGLE: ±{self.max_camera_angle:.1f}deg max range (randomized per episode reset, fixed during episode)")
        
        # 5. CAMERA NOISE PROGRESSION (D455 Simulation)
        initial_camera_gaussian_std, initial_camera_dropout_rate = self.task_config.curriculum.get_camera_noise(self.curriculum_level)
        logger.info(f"   5. CAMERA NOISE: Gaussian STD={initial_camera_gaussian_std:.4f}, Dropout={initial_camera_dropout_rate*100:.1f}% (both drone & static)")
        
        logger.info(f"   6. ASSET MANAGER: Updated both obs_dict and global_tensor_dict with count {total_obstacles_in_env}")
        
        # Calculate progress fraction
        self.curriculum_progress_fraction = (
            self.curriculum_level - self.task_config.curriculum.min_level
        ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)
        
        logger.info(f"   7. PROGRESS: {self.curriculum_progress_fraction:.3f} (level {self.curriculum_level}/{self.task_config.curriculum.max_level})")
        logger.info(f"   8. EVALUATION: Check every {self.task_config.curriculum.check_after_log_instances} instances (success rate threshold: {self.task_config.curriculum.success_rate_for_increase:.3f})")
        
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
        
        # GATE SCALING: Track current gate scale for each environment
        self.current_gate_scale = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.current_gate_tolerance = {
            'width': torch.ones(self.num_envs, dtype=torch.float32, device=self.device) * 1.3,
            'height_min': torch.ones(self.num_envs, dtype=torch.float32, device=self.device) * 0.2,
            'height_max': torch.ones(self.num_envs, dtype=torch.float32, device=self.device) * 2.2,
        }
        
        # Track which gate instance is active for each environment
        self.active_gate_instance = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Gate asset indices (will be populated during environment initialization)
        self.gate_asset_indices = {
            'full': None,    # Full size gate indices
            'medium': None,  # Medium size gate indices  
            'small': None,   # Small size gate indices
            'minimum': None  # Minimum size gate indices
        }
        
        # Initialize gate asset indices after environment is created
        logger.warning("[GATE SCALING INIT] Starting gate asset index discovery...")
        self._discover_gate_asset_indices()
        logger.warning("[GATE SCALING INIT] Gate asset index discovery completed")
        
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
        Reset environment instance at the specified indices.
        
        ⚠️ EMERGENCY GATE SCALING DISABLE:
        If NaN errors persist, set EMERGENCY_DISABLE_GATE_SCALING = True below 
        to immediately disable all gate scaling functionality and use default behavior.
        """
        # 🚨 EMERGENCY DISABLE FLAG - Set to True to disable gate scaling immediately
        EMERGENCY_DISABLE_GATE_SCALING = False  # ⚠️ Set to True to disable gate scaling
        
        if EMERGENCY_DISABLE_GATE_SCALING:
            logger.warning("🚨 EMERGENCY: Gate scaling DISABLED via emergency flag in reset_idx")
            logger.warning("🚨 EMERGENCY: Using original single gate behavior")
            # Call environment reset without gate scaling
            self.sim_env.reset_idx(env_ids)
            return
        
        logger.warning(f"[RESET DEBUG] reset_idx called for {len(env_ids)} environments: {env_ids}")
        logger.warning(f"[RESET DEBUG] Current curriculum level: {self.curriculum_level}")
        
        # Get scales for resetting environments for debugging
        current_scales = []
        for env_idx in env_ids:
            env_idx_int = env_idx.item()
            if env_idx_int < len(self.current_gate_scale):
                current_scales.append(f"{self.current_gate_scale[env_idx_int]:.1f}")
        logger.warning(f"[RESET DEBUG] Current scales for resetting envs: {current_scales}")
        
        # Apply curriculum-based gate scaling BEFORE environment reset
        self._apply_curriculum_gate_scaling(env_ids)
        
        # Call the environment reset method (NOT super().reset_idx which is abstract)
        self.sim_env.reset_idx(env_ids)
        
        # CRITICAL: Restore missing functionality that was removed during edit
        
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
        
        # RESET EPISODE REWARD TRACKING: Store completed episode data and reset trackers
        self.reset_episode_reward_tracking(env_ids)
        
        # Update static camera position based on curriculum level (ONLY for resetting environments)
        if len(env_ids) > 0:
            self.static_camera_manager.update_camera_positions(self.curriculum_level, env_ids)
            logger.debug(f"Updated static camera angles for {len(env_ids)} resetting environments: {env_ids.tolist()}")
        
        # DEBUGGING: Log final scales after reset
        if len(env_ids) > 0:
            final_scales = []
            for env_idx in env_ids:
                env_idx_int = env_idx.item()
                if env_idx_int < len(self.current_gate_scale):
                    final_scales.append(f"{self.current_gate_scale[env_idx_int]:.1f}")
            logger.warning(f"[RESET DEBUG] Final scales after reset: {final_scales}")
            
            # Show tolerance values that were applied
            try:
                avg_width_tol = torch.mean(self.current_gate_tolerance['width'][env_ids]).item()
                avg_height_range = torch.mean(self.current_gate_tolerance['height_max'][env_ids] - self.current_gate_tolerance['height_min'][env_ids]).item()
                logger.warning(f"[RESET DEBUG] Average tolerance applied: ±{avg_width_tol:.2f}m width, {avg_height_range:.1f}m height range")
            except Exception as e:
                logger.warning(f"[RESET DEBUG] Could not calculate tolerance averages: {e}")

        self.infos = {}
    
    def _discover_gate_asset_indices(self):
        """
        Discover the asset indices for different gate types.
        
        This method attempts to map gate asset types to their indices in the asset state tensor
        by examining the environment configuration and asset loading order.
        """
        try:
            # Check if we can access the environment configuration through asset_loader
            if not hasattr(self.sim_env, 'asset_loader'):
                logger.warning("[GATE SCALING] Asset loader not available for asset index discovery")
                logger.warning(f"[GATE SCALING] sim_env attributes: {dir(self.sim_env)}")
                return
                
            asset_loader = self.sim_env.asset_loader
            logger.warning(f"[GATE SCALING] Found asset_loader: {type(asset_loader)}")
            
            # Try to get environment config from asset_loader
            env_config = None
            if hasattr(asset_loader, 'env_config'):
                env_config = asset_loader.env_config
                logger.warning(f"[GATE SCALING] Found env_config in asset_loader: {type(env_config)}")
            elif hasattr(asset_loader, 'cfg'):
                env_config = asset_loader.cfg  
                logger.warning(f"[GATE SCALING] Found cfg in asset_loader: {type(env_config)}")
            else:
                logger.warning(f"[GATE SCALING] asset_loader attributes: {dir(asset_loader)}")
                logger.warning("[GATE SCALING] No env_config found in asset_loader")
                return
            
            # Look for asset type mapping
            if hasattr(env_config, 'asset_type_to_dict_map') and hasattr(env_config, 'include_asset_type'):
                asset_type_map = env_config.asset_type_to_dict_map
                include_assets = env_config.include_asset_type
                
                # Try to get actual asset loading information
                gate_indices = {}
                
                # Check if we can access the global_asset_dicts which contains the actual loaded assets
                if hasattr(self.sim_env, 'global_asset_dicts') and self.sim_env.global_asset_dicts:
                    logger.warning(f"[GATE SCALING] Found global_asset_dicts with {len(self.sim_env.global_asset_dicts)} environments")
                    
                    # Use the first environment's asset list as reference (all envs should have same asset types)
                    env_0_assets = self.sim_env.global_asset_dicts[0]
                    logger.warning(f"[GATE SCALING] Environment 0 has {len(env_0_assets)} assets")
                    
                    # First pass: collect ONLY specific gate assets (filter out legacy 'gate' type)
                    gate_assets = []
                    for asset_idx, asset_info in enumerate(env_0_assets):
                        asset_type = asset_info.get('asset_type', 'unknown')
                        filename = asset_info.get('filename', 'unknown')
                        logger.warning(f"[GATE SCALING] Asset {asset_idx}: type='{asset_type}', file='{filename}'")
                        
                        # Collect ONLY specific gate asset types (exclude legacy 'gate' type)
                        if asset_type in ['gate_full', 'gate_medium', 'gate_small', 'gate_minimum']:
                            gate_assets.append((asset_idx, asset_type))
                            logger.warning(f"[GATE SCALING] ✅ Collected specific gate asset: {asset_idx} ({asset_type})")
                        elif asset_type == 'gate':
                            # Skip legacy 'gate' type entirely
                            logger.warning(f"[GATE SCALING] ⚠️  Skipping legacy gate asset {asset_idx} (type='gate') - use specific gate types instead")
                    
                    logger.warning(f"[GATE SCALING] Found {len(gate_assets)} specific gate assets: {gate_assets}")
                    
                    # Ensure we have exactly 4 gate assets
                    if len(gate_assets) != 4:
                        logger.warning(f"[GATE SCALING] ❌ Expected exactly 4 specific gate assets, but found {len(gate_assets)}")
                        logger.warning(f"[GATE SCALING] Available assets: {gate_assets}")
                        logger.warning(f"[GATE SCALING] Check environment configuration - should include gate_full, gate_medium, gate_small, gate_minimum")
                    
                    # Map gate assets by their specific types (guaranteed to be unique)
                    logger.warning(f"[GATE SCALING] Starting asset mapping...")
                    filename_mapped = True  # We only accept specific types, so this is always successful
                    
                    for asset_idx, asset_type in gate_assets:
                        if asset_type == 'gate_full':
                            gate_indices['full'] = gate_indices.get('full', []) + [asset_idx]
                            logger.warning(f"[GATE SCALING] 🔵 Mapped asset {asset_idx} -> 'full' (by type)")
                        elif asset_type == 'gate_medium':
                            gate_indices['medium'] = gate_indices.get('medium', []) + [asset_idx]
                            logger.warning(f"[GATE SCALING] 🟢 Mapped asset {asset_idx} -> 'medium' (by type)")
                        elif asset_type == 'gate_small':
                            gate_indices['small'] = gate_indices.get('small', []) + [asset_idx]
                            logger.warning(f"[GATE SCALING] 🟠 Mapped asset {asset_idx} -> 'small' (by type)")
                        elif asset_type == 'gate_minimum':
                            gate_indices['minimum'] = gate_indices.get('minimum', []) + [asset_idx]
                            logger.warning(f"[GATE SCALING] 🔴 Mapped asset {asset_idx} -> 'minimum' (by type)")
                    
                    # Verify we have exactly 4 distinct gate types
                    gate_count = sum(1 for indices in gate_indices.values() if indices)
                    logger.warning(f"[GATE SCALING] ✅ Successfully mapped {gate_count}/4 gate types")
                    
                    if gate_count != 4:
                        logger.warning(f"[GATE SCALING] ❌ Expected 4 gate types, but mapped {gate_count}")
                        logger.warning(f"[GATE SCALING] Available gate types: {list(k for k, v in gate_indices.items() if v)}")
                        logger.warning(f"[GATE SCALING] Missing gate types: {list(k for k, v in gate_indices.items() if not v)}")
                    
                    # No fallback needed - we require specific gate types
                
                else:
                    # Fallback: Build mapping based on configuration order
                    logger.warning("[GATE SCALING] Using fallback: mapping based on config order")
                    asset_index = 0
                    
                    for asset_type, asset_config in asset_type_map.items():
                        # Skip if this asset type is not included
                        if asset_type in include_assets and not include_assets[asset_type]:
                            continue
                        
                        # Check if this is a gate asset
                        if asset_type.startswith('gate'):
                            num_assets = getattr(asset_config, 'num_assets', 1)
                            
                            # Map asset type to indices
                            if asset_type == 'gate_full':
                                gate_indices['full'] = list(range(asset_index, asset_index + num_assets))
                            elif asset_type == 'gate_medium':
                                gate_indices['medium'] = list(range(asset_index, asset_index + num_assets))
                            elif asset_type == 'gate_small':
                                gate_indices['small'] = list(range(asset_index, asset_index + num_assets))
                            elif asset_type == 'gate_minimum':
                                gate_indices['minimum'] = list(range(asset_index, asset_index + num_assets))
                            elif asset_type == 'gate':  # Original gate
                                gate_indices['full'] = gate_indices.get('full', []) + list(range(asset_index, asset_index + num_assets))
                            
                            asset_index += num_assets
                        else:
                            # Non-gate asset
                            num_assets = getattr(asset_config, 'num_assets', 1)
                            asset_index += num_assets
                
                # Update our gate asset indices
                for gate_type, indices in gate_indices.items():
                    if indices:
                        self.gate_asset_indices[gate_type] = torch.tensor(indices, device=self.device)
                        logger.warning(f"[GATE SCALING] Discovered {gate_type} gate indices: {indices}")
                
                # Log summary
                discovered = [k for k, v in self.gate_asset_indices.items() if v is not None]
                logger.warning(f"[GATE SCALING] Discovered gate types: {discovered}")
                
            else:
                logger.warning("[GATE SCALING] Asset configuration structure not as expected")
                
        except Exception as e:
            logger.warning(f"[GATE SCALING] Failed to discover gate asset indices: {e}")
            # Fallback: assume gates are the first few assets
            logger.warning("[GATE SCALING] Using fallback gate indexing (may not work correctly)")
            
            # Simple fallback: assume first 4 assets are the gates in order
            try:
                self.gate_asset_indices['full'] = torch.tensor([0], device=self.device)
                self.gate_asset_indices['medium'] = torch.tensor([1], device=self.device)
                self.gate_asset_indices['small'] = torch.tensor([2], device=self.device)
                self.gate_asset_indices['minimum'] = torch.tensor([3], device=self.device)
                logger.warning("[GATE SCALING] Applied fallback indexing: full=0, medium=1, small=2, minimum=3")
            except Exception as fallback_error:
                logger.warning(f"[GATE SCALING] Fallback indexing also failed: {fallback_error}")
    
    def _apply_curriculum_gate_scaling(self, env_ids):
        """
        Apply curriculum-based gate scaling for specified environments.
        
        ⚠️ EMERGENCY DISABLE INSTRUCTIONS:
        If NaN errors persist, set DISABLE_GATE_SCALING = True below to immediately disable
        gate scaling and revert to single default gate behavior.
        
        For each resetting environment:
        1. Select gate scale based on curriculum level
        2. Hide all gate instances off-screen
        3. Position the selected gate at environment center
        4. Update success tolerance for the selected gate scale
        
        Args:
            env_ids: Tensor of environment IDs that are resetting
        """
        # CRITICAL: Emergency disable flag for gate scaling (set to True to disable)
        DISABLE_GATE_SCALING = False  # ⚠️ Set to True if NaN errors persist
        if DISABLE_GATE_SCALING:
            logger.warning("[GATE SCALING] ⚠️ Gate scaling is DISABLED via emergency flag")
            logger.warning("[GATE SCALING] ⚠️ Using default gate configuration")
            return
        
        if len(env_ids) == 0:
            return
        
        # CRITICAL: Early validation - ensure environment is properly initialized
        logger.warning(f"[GATE SCALING] Performing early validation before gate scaling...")
        try:
            # Check if basic environment components are available
            if not hasattr(self, 'sim_env') or self.sim_env is None:
                logger.warning(f"[GATE SCALING] ❌ CRITICAL: sim_env not available - aborting gate scaling")
                return
                
            if not hasattr(self.sim_env, 'asset_manager') or self.sim_env.asset_manager is None:
                logger.warning(f"[GATE SCALING] ❌ CRITICAL: asset_manager not available - aborting gate scaling")
                return
                
            if not hasattr(self.sim_env.asset_manager, 'env_asset_state_tensor'):
                logger.warning(f"[GATE SCALING] ❌ CRITICAL: env_asset_state_tensor not available - aborting gate scaling")
                return
                
            # Check if observation dict is properly initialized
            if not hasattr(self, 'obs_dict') or self.obs_dict is None:
                logger.warning(f"[GATE SCALING] ❌ CRITICAL: obs_dict not available - aborting gate scaling")
                return
                
            if "env_bounds_min" not in self.obs_dict or "env_bounds_max" not in self.obs_dict:
                logger.warning(f"[GATE SCALING] ❌ CRITICAL: environment bounds not available - aborting gate scaling")
                return
                
            # Validate tensor shapes and basic integrity
            asset_state_tensor = self.sim_env.asset_manager.env_asset_state_tensor
            if asset_state_tensor is None:
                logger.warning(f"[GATE SCALING] ❌ CRITICAL: asset_state_tensor is None - aborting gate scaling")
                return
                
            if len(asset_state_tensor.shape) != 3:
                logger.warning(f"[GATE SCALING] ❌ CRITICAL: asset_state_tensor has invalid shape {asset_state_tensor.shape} - aborting gate scaling")
                return
                
            # Check for basic tensor corruption
            if torch.isnan(asset_state_tensor).any():
                logger.warning(f"[GATE SCALING] ❌ CRITICAL: asset_state_tensor contains NaN values before gate scaling - aborting")
                return
                
            if torch.isinf(asset_state_tensor).any():
                logger.warning(f"[GATE SCALING] ❌ CRITICAL: asset_state_tensor contains infinite values before gate scaling - aborting")
                return
                
            logger.warning(f"[GATE SCALING] ✅ Early validation passed - environment ready for gate scaling")
            
        except Exception as e:
            logger.warning(f"[GATE SCALING] ❌ CRITICAL: Early validation failed: {e}")
            logger.warning(f"[GATE SCALING] ❌ Aborting gate scaling to prevent corruption")
            import traceback
            logger.warning(f"[GATE SCALING] Error traceback: {traceback.format_exc()}")
            return
        
        # Check if asset indices were discovered, if not try again
        if all(indices is None for indices in self.gate_asset_indices.values()):
            logger.warning("[GATE SCALING] Asset indices not discovered yet, retrying discovery...")
            self._discover_gate_asset_indices()
            
        # Get current curriculum level
        current_level = self.curriculum_level
        
        logger.warning(f"[GATE SCALING] Applying scaling for curriculum level {current_level} to {len(env_ids)} environments")
        
        # Select gate scales for each resetting environment
        scales_applied = []
        for env_idx in env_ids:
            env_idx_int = env_idx.item()
            
            # Get gate scale for current curriculum level (with randomization)
            try:
                selected_scale = self.task_config.curriculum.get_gate_scale_for_level(current_level)
                scales_applied.append(selected_scale)
                logger.warning(f"[GATE SCALING] Env {env_idx_int}: Selected scale {selected_scale}")
            except Exception as e:
                logger.warning(f"[GATE SCALING ERROR] Failed to get scale for level {current_level}: {e}")
                selected_scale = 1.0  # Fallback to full size
                scales_applied.append(selected_scale)
            
            # DEBUG: Log asset indices info occasionally
            if env_idx_int == 0:  # Only log for environment 0 to avoid spam
                available_indices = {k: v for k, v in self.gate_asset_indices.items() if v is not None}
                if available_indices:
                    logger.warning(f"[GATE SCALING] Available gate asset indices: {available_indices}")
                else:
                    logger.warning(f"[GATE SCALING] No gate asset indices available for positioning")
            
            # Update tracking variables
            old_scale = self.current_gate_scale[env_idx_int].item()
            self.current_gate_scale[env_idx_int] = selected_scale
            
            if old_scale != selected_scale:
                logger.debug(f"[GATE SCALING] Env {env_idx_int}: Scale changed {old_scale:.1f} -> {selected_scale:.1f}")
            
            # Get adaptive tolerance for this scale
            width_tolerance, height_min, height_max = self.task_config.curriculum.get_gate_tolerance_for_scale(selected_scale)
            self.current_gate_tolerance['width'][env_idx_int] = width_tolerance
            self.current_gate_tolerance['height_min'][env_idx_int] = height_min
            self.current_gate_tolerance['height_max'][env_idx_int] = height_max
            
            # Map scale to gate instance type
            if selected_scale >= 1.0:
                gate_type = 'full'
                instance_id = 0
            elif selected_scale >= 0.7:
                gate_type = 'medium'
                instance_id = 1
            elif selected_scale >= 0.5:
                gate_type = 'small'
                instance_id = 2
            else:
                gate_type = 'minimum'
                instance_id = 3
            
            self.active_gate_instance[env_idx_int] = instance_id
            
            logger.warning(f"[GATE SCALING] Env {env_idx_int}: Mapping scale {selected_scale} -> gate type '{gate_type}' (instance {instance_id})")
            
            # Position the selected gate at environment center and hide others
            logger.warning(f"[GATE SCALING] About to call _position_gate_instances for env {env_idx_int}")
            self._position_gate_instances(env_idx_int, gate_type, selected_scale)
        
        # Log gate scaling information
        if len(env_ids) > 0:
            unique_scales = torch.unique(self.current_gate_scale[env_ids])
            scale_str = ", ".join([f"{scale:.1f}" for scale in unique_scales])
            logger.debug(f"[GATE SCALING] Applied gate scaling for {len(env_ids)} environments. Scales used: {scale_str}")
            
            # Count scales applied this reset
            from collections import Counter
            scale_counts = Counter(scales_applied)
            scale_summary = ", ".join([f"{scale:.1f}x{count}" for scale, count in sorted(scale_counts.items(), reverse=True)])
            logger.debug(f"[GATE SCALING] Scale distribution this reset: {scale_summary}")
            
            # Show available scales for this level for comparison
            try:
                from aerial_gym.config.asset_config.gate_scaling_config import GateScalingConfig
                available_scales = GateScalingConfig.get_available_scales_for_level(current_level)
                logger.debug(f"[GATE SCALING] Available scales for level {current_level}: {available_scales}")
            except Exception as e:
                logger.warning(f"[GATE SCALING] Could not get available scales: {e}")
        
        # CRITICAL: Apply all gate positioning changes to Isaac Gym simulation
        # This is called once after all environments have been processed for efficiency
        logger.warning(f"[GATE SCALING] Applying all gate positioning changes to Isaac Gym simulation...")
        
        # CRITICAL: Validate tensor integrity before write_to_sim
        logger.warning(f"[GATE SCALING] Performing critical tensor validation before write_to_sim...")
        try:
            asset_state_tensor = self.sim_env.asset_manager.env_asset_state_tensor
            
            # Check for NaN values in the entire tensor
            if torch.isnan(asset_state_tensor).any():
                logger.warning(f"[GATE SCALING] ❌ CRITICAL: Asset state tensor contains NaN values!")
                logger.warning(f"[GATE SCALING] ❌ ABORTING write_to_sim to prevent simulation corruption!")
                logger.warning(f"[GATE SCALING] ❌ Gate positioning has been completed but NOT applied to Isaac Gym")
                logger.warning(f"[GATE SCALING] ❌ This will maintain current gate positions but prevent NaN propagation")
                return
            
            # Check for infinite values
            if torch.isinf(asset_state_tensor).any():
                logger.warning(f"[GATE SCALING] ❌ CRITICAL: Asset state tensor contains infinite values!")
                logger.warning(f"[GATE SCALING] ❌ ABORTING write_to_sim to prevent simulation corruption!")
                return
            
            # Check for extremely large values that could cause issues
            max_pos_magnitude = torch.abs(asset_state_tensor[:, :, 0:3]).max()
            if max_pos_magnitude > 100.0:
                logger.warning(f"[GATE SCALING] ⚠️ WARNING: Very large position values detected (max: {max_pos_magnitude})")
                logger.warning(f"[GATE SCALING] ⚠️ This might indicate positioning errors")
            
            logger.warning(f"[GATE SCALING] ✅ Tensor validation passed - proceeding with write_to_sim")
            
        except Exception as e:
            logger.warning(f"[GATE SCALING] ❌ CRITICAL: Failed to validate tensors: {e}")
            logger.warning(f"[GATE SCALING] ❌ ABORTING write_to_sim due to validation failure")
            return
        
        try:
            # Fix #1: Corrected Isaac Gym Access Path
            # Before: self.sim_env.write_to_sim() ❌ (doesn't exist)
            # After: self.sim_env.IGE_env.write_to_sim() ✅ (correct path)
            
            # Fix #3: Added Comprehensive Debug Tracking
            logger.warning(f"[GATE SCALING] Analyzing Isaac Gym environment structure...")
            logger.warning(f"[GATE SCALING] sim_env type: {type(self.sim_env)}")
            logger.warning(f"[GATE SCALING] sim_env has IGE_env: {hasattr(self.sim_env, 'IGE_env')}")
            
            if hasattr(self.sim_env, 'IGE_env'):
                logger.warning(f"[GATE SCALING] IGE_env type: {type(self.sim_env.IGE_env)}")
                logger.warning(f"[GATE SCALING] IGE_env has write_to_sim: {hasattr(self.sim_env.IGE_env, 'write_to_sim')}")
                
                # Correct path to Isaac Gym environment
                if hasattr(self.sim_env.IGE_env, 'write_to_sim'):
                    logger.warning(f"[GATE SCALING] Calling IGE_env.write_to_sim()...")
                    self.sim_env.IGE_env.write_to_sim()
                    logger.warning(f"[GATE SCALING] ✅ Successfully applied changes to Isaac Gym via IGE_env.write_to_sim()")
                    
                    # CRITICAL: Validate tensor integrity after write_to_sim
                    logger.warning(f"[GATE SCALING] Validating tensor integrity after write_to_sim...")
                    if torch.isnan(asset_state_tensor).any():
                        logger.warning(f"[GATE SCALING] ❌ CRITICAL: Asset state tensor corrupted after write_to_sim!")
                        logger.warning(f"[GATE SCALING] ❌ This indicates a serious issue with tensor synchronization")
                    else:
                        logger.warning(f"[GATE SCALING] ✅ Tensor integrity maintained after write_to_sim")
                        
                else:
                    # Fallback: Access Isaac Gym directly via IGE_env
                    logger.warning(f"[GATE SCALING] Using direct Isaac Gym tensor access via IGE_env...")
                    import gymtorch
                    if hasattr(self.sim_env.IGE_env, 'global_tensor_dict'):
                        self.sim_env.IGE_env.gym.set_actor_root_state_tensor(
                            self.sim_env.IGE_env.sim,
                            gymtorch.unwrap_tensor(self.sim_env.IGE_env.global_tensor_dict["unfolded_env_asset_state_tensor"]),
                        )
                        logger.warning(f"[GATE SCALING] ✅ Successfully applied changes to Isaac Gym via direct IGE_env call")
                    else:
                        logger.warning(f"[GATE SCALING] ❌ IGE_env missing global_tensor_dict")
                        raise Exception("Cannot access global_tensor_dict in IGE_env")
                
                # Force immediate graphics update for visual feedback  
                if hasattr(self.sim_env.IGE_env, 'step_graphics'):
                    logger.warning(f"[GATE SCALING] Calling IGE_env.step_graphics()...")
                    self.sim_env.IGE_env.step_graphics()
                    logger.warning(f"[GATE SCALING] ✅ Updated graphics for immediate visual feedback via IGE_env")
                else:
                    logger.warning(f"[GATE SCALING] ⚠️ IGE_env.step_graphics() not available")
                    
            elif hasattr(self.sim_env, 'write_to_sim'):
                # Fallback to original path (shouldn't happen based on logs)
                logger.warning(f"[GATE SCALING] Using fallback write_to_sim()...")
                self.sim_env.write_to_sim()
                logger.warning(f"[GATE SCALING] ✅ Successfully applied changes to Isaac Gym via fallback write_to_sim()")
            else:
                # Final fallback: Try to access through global_tensor_dict directly
                logger.warning(f"[GATE SCALING] Using final fallback - direct tensor access...")
                import gymtorch
                
                # Try multiple paths to find the correct tensor dict
                tensor_dict = None
                if hasattr(self.sim_env, 'global_tensor_dict'):
                    tensor_dict = self.sim_env.global_tensor_dict
                    logger.warning(f"[GATE SCALING] Found tensor dict via sim_env.global_tensor_dict")
                elif hasattr(self.sim_env, 'IGE_env') and hasattr(self.sim_env.IGE_env, 'global_tensor_dict'):
                    tensor_dict = self.sim_env.IGE_env.global_tensor_dict
                    logger.warning(f"[GATE SCALING] Found tensor dict via sim_env.IGE_env.global_tensor_dict")
                    
                if tensor_dict and "unfolded_env_asset_state_tensor" in tensor_dict:
                    # Try to find gym and sim objects
                    gym_obj = None
                    sim_obj = None
                    
                    if hasattr(self.sim_env, 'gym'):
                        gym_obj = self.sim_env.gym
                        sim_obj = self.sim_env.sim
                    elif hasattr(self.sim_env, 'IGE_env'):
                        gym_obj = self.sim_env.IGE_env.gym
                        sim_obj = self.sim_env.IGE_env.sim
                        
                    if gym_obj and sim_obj:
                        gym_obj.set_actor_root_state_tensor(
                            sim_obj,
                            gymtorch.unwrap_tensor(tensor_dict["unfolded_env_asset_state_tensor"]),
                        )
                        logger.warning(f"[GATE SCALING] ✅ Successfully applied changes via final fallback")
                    else:
                        logger.warning(f"[GATE SCALING] ❌ Cannot find gym/sim objects")
                        raise Exception("Cannot find gym/sim objects for tensor update")
                else:
                    logger.warning(f"[GATE SCALING] ❌ Cannot find tensor dict or unfolded_env_asset_state_tensor")
                    raise Exception("Cannot find required tensors for update")
                
        except Exception as e:
            logger.warning(f"[GATE SCALING] ❌ Failed to apply changes to Isaac Gym: {e}")
            import traceback
            logger.warning(f"[GATE SCALING] Error traceback: {traceback.format_exc()}")
            
            # Additional debug info on failure
            logger.warning(f"[GATE SCALING] DEBUG INFO ON FAILURE:")
            logger.warning(f"[GATE SCALING]   sim_env attributes: {dir(self.sim_env)}")
            if hasattr(self.sim_env, 'IGE_env'):
                logger.warning(f"[GATE SCALING]   IGE_env attributes: {dir(self.sim_env.IGE_env)}")
    
    def _position_gate_instances(self, env_idx, active_gate_type, scale_factor):
        """
        Position gate instances for a specific environment.
        
        Shows the selected gate at environment center and hides all others off-screen.
        
        Args:
            env_idx: Environment index
            active_gate_type: Type of gate to show ('full', 'medium', 'small', 'minimum')
            scale_factor: Scale factor for logging purposes
        """
        logger.warning(f"[GATE POSITIONING] Starting positioning for env {env_idx}, gate type {active_gate_type}, scale {scale_factor}")
        
        if not hasattr(self.sim_env, 'asset_manager') or not hasattr(self.sim_env.asset_manager, 'env_asset_state_tensor'):
            logger.warning(f"[GATE POSITIONING] Asset manager not available for gate positioning in env {env_idx}")
            return
        
        # Check if we have discovered gate asset indices
        available_indices = {k: v for k, v in self.gate_asset_indices.items() if v is not None}
        if not available_indices:
            logger.warning(f"[GATE POSITIONING] No gate asset indices available for positioning")
            logger.warning(f"[GATE POSITIONING] Falling back to basic gate positioning")
            self._fallback_gate_positioning(env_idx, active_gate_type, scale_factor)
            return
        
        logger.warning(f"[GATE POSITIONING] Available indices: {available_indices}")
        
        try:
            # Access the asset state tensor
            logger.warning(f"[GATE POSITIONING] Accessing asset state tensor...")
            asset_state_tensor = self.sim_env.asset_manager.env_asset_state_tensor
            logger.warning(f"[GATE POSITIONING] Asset state tensor shape: {asset_state_tensor.shape}")
            
            # CRITICAL: Add NaN validation before any tensor operations
            logger.warning(f"[GATE POSITIONING] Validating tensor integrity before modifications...")
            if torch.isnan(asset_state_tensor).any():
                logger.warning(f"[GATE POSITIONING] ❌ CRITICAL: Asset state tensor contains NaN values before gate positioning!")
                logger.warning(f"[GATE POSITIONING] ❌ Aborting gate positioning to prevent corruption")
                return
            
            # Validate environment index bounds
            if env_idx < 0 or env_idx >= asset_state_tensor.shape[0]:
                logger.warning(f"[GATE POSITIONING] ❌ Invalid environment index {env_idx}, tensor has {asset_state_tensor.shape[0]} environments")
                return
            
            # Get environment bounds for this environment
            logger.warning(f"[GATE POSITIONING] Getting environment bounds for env {env_idx}...")
            env_bounds_min = self.obs_dict["env_bounds_min"][env_idx]
            env_bounds_max = self.obs_dict["env_bounds_max"][env_idx]
            logger.warning(f"[GATE POSITIONING] Env bounds: min={env_bounds_min}, max={env_bounds_max}")
            
            # CRITICAL: Validate environment bounds for NaN
            if torch.isnan(env_bounds_min).any() or torch.isnan(env_bounds_max).any():
                logger.warning(f"[GATE POSITIONING] ❌ CRITICAL: Environment bounds contain NaN values!")
                logger.warning(f"[GATE POSITIONING] ❌ Aborting gate positioning to prevent corruption")
                return
            
            # Calculate environment center (ensure valid values)
            env_center = (env_bounds_min + env_bounds_max) / 2.0
            env_center[2] = 0.0  # Set Z to ground level for gates
            logger.warning(f"[GATE POSITIONING] Calculated env center: {env_center}")
            
            # CRITICAL: Validate environment center for NaN
            if torch.isnan(env_center).any():
                logger.warning(f"[GATE POSITIONING] ❌ CRITICAL: Environment center contains NaN values!")
                logger.warning(f"[GATE POSITIONING] ❌ Aborting gate positioning to prevent corruption")
                return
            
            # Define off-screen position (far away, but not NaN)
            off_screen_pos = torch.tensor([-50.0, -50.0, 0.0], device=env_center.device, dtype=env_center.dtype)
            
            # Gate type processing order (consistent ordering)
            gate_type_names = ['full', 'medium', 'small', 'minimum']
            positioned_count = 0
            
            # Process each gate type
            for gate_type in gate_type_names:
                gate_indices = self.gate_asset_indices[gate_type]
                logger.warning(f"[GATE POSITIONING] Processing gate type '{gate_type}' with indices: {gate_indices}")
                
                if gate_indices is not None:
                    for i, asset_idx in enumerate(gate_indices):
                        logger.warning(f"[GATE POSITIONING] Processing asset index {asset_idx}...")
                        
                        # CRITICAL: Validate asset index bounds
                        if asset_idx < 0 or asset_idx >= asset_state_tensor.shape[1]:
                            logger.warning(f"[GATE POSITIONING] ❌ Invalid asset index {asset_idx}, tensor has {asset_state_tensor.shape[1]} assets")
                            continue
                        
                        # CRITICAL: Check current asset position for NaN before modification
                        current_pos = asset_state_tensor[env_idx, asset_idx, 0:3].clone()
                        if torch.isnan(current_pos).any():
                            logger.warning(f"[GATE POSITIONING] ❌ CRITICAL: Asset {asset_idx} position contains NaN before modification!")
                            logger.warning(f"[GATE POSITIONING] ❌ Skipping this asset to prevent further corruption")
                            continue
                        
                        if gate_type == active_gate_type and i == 0:
                            # Position ONLY THE FIRST active gate at environment center
                            logger.warning(f"[GATE POSITIONING] Setting active gate position...")
                            
                            # CRITICAL: Create a copy to ensure no NaN propagation
                            new_pos = env_center.clone()
                            if torch.isnan(new_pos).any():
                                logger.warning(f"[GATE POSITIONING] ❌ CRITICAL: New position contains NaN!")
                                continue
                                
                            asset_state_tensor[env_idx, asset_idx, 0:3] = new_pos
                            logger.warning(f"[GATE POSITIONING] ✅ Env {env_idx}: Positioned {gate_type} gate (idx {asset_idx}) at center {new_pos}")
                            positioned_count += 1
                            
                            # CRITICAL: Validate after modification
                            post_mod_pos = asset_state_tensor[env_idx, asset_idx, 0:3]
                            if torch.isnan(post_mod_pos).any():
                                logger.warning(f"[GATE POSITIONING] ❌ CRITICAL: Position became NaN after modification!")
                                # Restore original position
                                asset_state_tensor[env_idx, asset_idx, 0:3] = current_pos
                                logger.warning(f"[GATE POSITIONING] ✅ Restored original position to prevent corruption")
                                continue
                                
                        else:
                            # Hide inactive gates AND duplicate gates off-screen
                            if gate_type == active_gate_type and i > 0:
                                logger.warning(f"[GATE POSITIONING] Moving duplicate {gate_type} gate off-screen...")
                            else:
                                logger.warning(f"[GATE POSITIONING] Moving inactive gate off-screen...")
                            
                            # CRITICAL: Ensure off-screen position is valid
                            new_pos = off_screen_pos.clone()
                            if torch.isnan(new_pos).any():
                                logger.warning(f"[GATE POSITIONING] ❌ CRITICAL: Off-screen position contains NaN!")
                                continue
                                
                            asset_state_tensor[env_idx, asset_idx, 0:3] = new_pos
                            logger.warning(f"[GATE POSITIONING] ✅ Env {env_idx}: Moved {gate_type} gate (idx {asset_idx}) off-screen")
                            
                            # CRITICAL: Validate after modification
                            post_mod_pos = asset_state_tensor[env_idx, asset_idx, 0:3]
                            if torch.isnan(post_mod_pos).any():
                                logger.warning(f"[GATE POSITIONING] ❌ CRITICAL: Position became NaN after off-screen modification!")
                                # Restore original position
                                asset_state_tensor[env_idx, asset_idx, 0:3] = current_pos
                                logger.warning(f"[GATE POSITIONING] ✅ Restored original position to prevent corruption")
                                continue
            
            # Update gate position tracking for this environment
            logger.warning(f"[GATE POSITIONING] Updating gate position tracking...")
            
            # CRITICAL FIX: Update gate_position to match the ACTUAL active gate position
            # Previously: Always set to env_center [0, 0, 0] regardless of gate type
            # Now: Get the actual position of the active gate from the asset tensor
            try:
                # Find the active gate index for this gate type
                active_gate_indices = self.gate_asset_indices[active_gate_type]
                if active_gate_indices is not None and len(active_gate_indices) > 0:
                    # Get the position of the first (active) gate of this type
                    active_gate_idx = active_gate_indices[0]
                    actual_gate_position = asset_state_tensor[env_idx, active_gate_idx, 0:3].clone()
                    
                    # Validate the actual gate position
                    if torch.isnan(actual_gate_position).any():
                        logger.warning(f"[GATE POSITIONING] ❌ CRITICAL: Actual gate position contains NaN!")
                        logger.warning(f"[GATE POSITIONING] ❌ Using fallback environment center position")
                        self.gate_position[env_idx] = env_center
                    else:
                        # Use the actual gate position for reward calculations
                        self.gate_position[env_idx] = actual_gate_position
                        logger.warning(f"[GATE POSITIONING] ✅ Updated gate_position[{env_idx}] to actual gate position: {actual_gate_position}")
                else:
                    logger.warning(f"[GATE POSITIONING] ⚠️ No active gate indices found, using environment center")
                    if torch.isnan(env_center).any():
                        logger.warning(f"[GATE POSITIONING] ❌ CRITICAL: Environment center NaN during tracking update!")
                        # Use a safe fallback position
                        self.gate_position[env_idx] = torch.tensor([0.0, 0.0, 0.0], device=env_center.device, dtype=env_center.dtype)
                    else:
                        self.gate_position[env_idx] = env_center
                        
            except Exception as e:
                logger.warning(f"[GATE POSITIONING] ❌ Failed to update gate_position tracking: {e}")
                # Fallback to environment center
                if not torch.isnan(env_center).any():
                    self.gate_position[env_idx] = env_center
                else:
                    logger.warning(f"[GATE POSITIONING] ❌ CRITICAL: Both actual and fallback positions invalid!")
                    # Use absolute fallback
                    self.gate_position[env_idx] = torch.tensor([0.0, 0.0, 0.0], device=env_center.device, dtype=env_center.dtype)
            
            # CRITICAL: Final tensor validation
            logger.warning(f"[GATE POSITIONING] Performing final tensor validation...")
            if torch.isnan(asset_state_tensor).any():
                logger.warning(f"[GATE POSITIONING] ❌ CRITICAL: Asset state tensor contains NaN values after gate positioning!")
                logger.warning(f"[GATE POSITIONING] ❌ This indicates tensor corruption - investigation needed!")
            else:
                logger.warning(f"[GATE POSITIONING] ✅ Final tensor validation passed - no NaN values detected")
            
            # CRITICAL: Validate robot positions are not corrupted
            logger.warning(f"[GATE POSITIONING] Validating robot positions are not corrupted...")
            try:
                # Robot should be at index 0 in the tensor (before environment assets)
                if hasattr(self.sim_env, 'asset_manager') and hasattr(self.sim_env.asset_manager, 'robot_state_tensor'):
                    robot_positions = self.sim_env.asset_manager.robot_state_tensor[:, 0:3]
                    if torch.isnan(robot_positions).any():
                        logger.warning(f"[GATE POSITIONING] ❌ CRITICAL: Robot positions contain NaN values!")
                        logger.warning(f"[GATE POSITIONING] ❌ Gate positioning may have corrupted robot states")
                    else:
                        logger.warning(f"[GATE POSITIONING] ✅ Robot positions are valid - no corruption detected")
                elif hasattr(self, 'obs_dict') and 'robot_position' in self.obs_dict:
                    robot_positions = self.obs_dict['robot_position']
                    if torch.isnan(robot_positions).any():
                        logger.warning(f"[GATE POSITIONING] ❌ CRITICAL: Robot positions in obs_dict contain NaN values!")
                    else:
                        logger.warning(f"[GATE POSITIONING] ✅ Robot positions in obs_dict are valid")
                else:
                    logger.warning(f"[GATE POSITIONING] ⚠️ Could not validate robot positions - no access path found")
            except Exception as e:
                logger.warning(f"[GATE POSITIONING] ⚠️ Failed to validate robot positions: {e}")
            
            logger.warning(f"[GATE POSITIONING] ✅ SUCCESS: Env {env_idx} positioned {positioned_count} gates for {active_gate_type} gate (scale {scale_factor:.1f})")
            
        except Exception as e:
            logger.warning(f"[GATE POSITIONING] Failed to position gate instances for env {env_idx}: {e}")
            import traceback
            logger.debug(f"[GATE POSITIONING] Error traceback: {traceback.format_exc()}")
    
    def _fallback_gate_positioning(self, env_idx, active_gate_type, scale_factor):
        """
        Fallback gate positioning when multiple gate instances are not available.
        
        This method works with a single gate asset and just updates the tracking variables.
        The actual scaling happens through the tolerance adjustments.
        
        Args:
            env_idx: Environment index
            active_gate_type: Type of gate ('full', 'medium', 'small', 'minimum')  
            scale_factor: Scale factor for the gate
        """
        try:
            logger.warning(f"[GATE POSITIONING FALLBACK] Positioning single gate for env {env_idx}")
            
            # Get environment bounds for this environment
            env_bounds_min = self.obs_dict["env_bounds_min"][env_idx]
            env_bounds_max = self.obs_dict["env_bounds_max"][env_idx]
            
            # Calculate environment center position (where the gate should be)
            env_center = (env_bounds_min + env_bounds_max) / 2.0
            env_center[2] = 0.0  # Place gate at ground level
            
            # Update gate position tracking for this environment
            self.gate_position[env_idx] = env_center
            
            logger.warning(f"[GATE POSITIONING FALLBACK] Env {env_idx}: Set gate position to {env_center} for {active_gate_type} gate (scale {scale_factor:.1f})")
            logger.warning(f"[GATE POSITIONING FALLBACK] Note: Physical gate scaling handled through tolerance adjustment only")
            
            # Try to position the gate asset if we can find it
            if hasattr(self.sim_env, 'asset_manager') and hasattr(self.sim_env.asset_manager, 'env_asset_state_tensor'):
                asset_state_tensor = self.sim_env.asset_manager.env_asset_state_tensor
                
                # Look for any gate-like asset (could be at any index)
                # For now, assume the gate is one of the first few assets
                for potential_gate_idx in range(min(5, asset_state_tensor.shape[1])):
                    try:
                        # Set position of this potential gate asset
                        asset_state_tensor[env_idx, potential_gate_idx, 0:3] = env_center
                        logger.warning(f"[GATE POSITIONING FALLBACK] Env {env_idx}: Positioned asset {potential_gate_idx} at center")
                        break  # Only position the first asset we find
                    except Exception as e:
                        logger.debug(f"[GATE POSITIONING FALLBACK] Could not position asset {potential_gate_idx}: {e}")
                        continue
            
        except Exception as e:
            logger.warning(f"[GATE POSITIONING FALLBACK] Failed for env {env_idx}: {e}")
            import traceback
            logger.debug(f"[GATE POSITIONING FALLBACK] Error traceback: {traceback.format_exc()}")
    
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
        
        # ADAPTIVE GATE PASSAGE DETECTION: Use curriculum-based tolerances
        # Tolerances automatically scale with gate size (full: ±1.3m, minimum: ±0.52m)
        
        # DEBUG: Log tolerance info occasionally
        if self.num_task_steps % 1000 == 0:  # Every 1000 steps
            avg_width_tol = torch.mean(self.current_gate_tolerance['width']).item()
            avg_height_min = torch.mean(self.current_gate_tolerance['height_min']).item()
            avg_height_max = torch.mean(self.current_gate_tolerance['height_max']).item()
            avg_scale = torch.mean(self.current_gate_scale).item()
            logger.debug(f"[TOLERANCE DEBUG] Step {self.num_task_steps}: Avg scale {avg_scale:.2f}, width ±{avg_width_tol:.2f}m, height {avg_height_min:.1f}-{avg_height_max:.1f}m")
        
        gate_passage_success = (
            (robot_position[:, 1] > self.gate_position[:, 1]) &  # Crossed gate (Y > 0)
            (torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < self.current_gate_tolerance['width']) &  # Adaptive width tolerance
            (robot_position[:, 2] > self.current_gate_tolerance['height_min']) & 
            (robot_position[:, 2] < self.current_gate_tolerance['height_max'])  # Adaptive height range
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
        
        # Check if robot has passed gate (crossed Y = 0 plane with proper alignment)
        # Use adaptive tolerances based on current gate scale
        gate_passed_current = (
            (robot_position[:, 1] > self.gate_position[:, 1]) &  # In front of gate
            (torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < self.current_gate_tolerance['width']) &  # Adaptive width tolerance
            (robot_position[:, 2] > self.current_gate_tolerance['height_min']) & 
            (robot_position[:, 2] < self.current_gate_tolerance['height_max'])  # Adaptive height range
        )
        
        # Gate alignment: check if robot is roughly aligned with gate opening (adaptive tolerance)
        gate_alignment = torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < self.current_gate_tolerance['width']
        
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
        
        # Add gate scaling tracking for wandb
        self.infos["gate_scaling/average_scale"] = torch.mean(self.current_gate_scale)
        self.infos["gate_scaling/min_scale"] = torch.min(self.current_gate_scale)
        self.infos["gate_scaling/max_scale"] = torch.max(self.current_gate_scale)
        self.infos["gate_scaling/average_width_tolerance"] = torch.mean(self.current_gate_tolerance['width'])
        self.infos["gate_scaling/average_height_range"] = torch.mean(self.current_gate_tolerance['height_max'] - self.current_gate_tolerance['height_min'])

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
                        # If already tensor
                        if static_depth_noised.dim() == 2:
                            static_depth_noised = static_depth_noised.unsqueeze(0)
                        static_depth_expanded = static_depth_noised.expand(self.sim_env.num_envs, -1, -1)
                        
                        encoded_latents = self.shared_vae_model.encode(static_depth_expanded)
                        self.static_image_latents[:] = encoded_latents
                        
                except Exception as vae_error:
                    logger.error(f"❌ VAE encoding failed: {vae_error}")
                    # Fallback to zeros if VAE fails
                    self.static_image_latents.fill_(0.0)
                    
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

    def compute_rewards_and_crashes(self, obs_dict):
        """Compute rewards with gate-specific components."""
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        
        # CRITICAL: Add NaN validation for inputs to prevent downstream corruption
        logger.debug(f"[REWARD CALC] Validating inputs for reward calculation...")
        
        # Validate robot position
        if torch.isnan(robot_position).any():
            logger.warning(f"[REWARD CALC] ❌ CRITICAL: robot_position contains NaN values!")
            logger.warning(f"[REWARD CALC] ❌ Using safe fallback positions")
            robot_position = torch.zeros_like(robot_position)
            
        # Validate target position  
        if torch.isnan(target_position).any():
            logger.warning(f"[REWARD CALC] ❌ CRITICAL: target_position contains NaN values!")
            logger.warning(f"[REWARD CALC] ❌ Using safe fallback target")
            target_position = torch.zeros_like(target_position)
            
        # Validate gate position
        if torch.isnan(self.gate_position).any():
            logger.warning(f"[REWARD CALC] ❌ CRITICAL: gate_position contains NaN values!")
            logger.warning(f"[REWARD CALC] ❌ Using safe fallback gate positions")
            self.gate_position = torch.zeros_like(self.gate_position)
            
        # Validate robot orientation
        if torch.isnan(robot_vehicle_orientation).any():
            logger.warning(f"[REWARD CALC] ❌ CRITICAL: robot_vehicle_orientation contains NaN values!")
            logger.warning(f"[REWARD CALC] ❌ Using safe fallback orientation")
            # Create identity quaternions [0, 0, 0, 1]
            robot_vehicle_orientation = torch.zeros_like(robot_vehicle_orientation)
            robot_vehicle_orientation[:, 3] = 1.0  # w component of identity quaternion
        
        self.pos_error_vehicle_frame_prev[:] = self.pos_error_vehicle_frame
        self.pos_error_vehicle_frame[:] = quat_rotate_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        
        # CRITICAL: Validate pos_error calculations
        if torch.isnan(self.pos_error_vehicle_frame).any():
            logger.warning(f"[REWARD CALC] ❌ CRITICAL: pos_error_vehicle_frame contains NaN values after calculation!")
            logger.warning(f"[REWARD CALC] ❌ Using safe fallback error values")
            self.pos_error_vehicle_frame = torch.zeros_like(self.pos_error_vehicle_frame)
            
        if torch.isnan(self.pos_error_vehicle_frame_prev).any():
            logger.warning(f"[REWARD CALC] ❌ CRITICAL: pos_error_vehicle_frame_prev contains NaN values!")
            logger.warning(f"[REWARD CALC] ❌ Using safe fallback previous error values")
            self.pos_error_vehicle_frame_prev = torch.zeros_like(self.pos_error_vehicle_frame_prev)
        
        # CRITICAL FIX: Clone action tensors to break reference dependency
        # obs_dict contains direct references to global tensors that get updated simultaneously
        current_actions = obs_dict["robot_actions"].clone()
        previous_actions = obs_dict["robot_prev_actions"].clone()
        
        # CRITICAL: Validate action tensors
        if torch.isnan(current_actions).any():
            logger.warning(f"[REWARD CALC] ❌ CRITICAL: current_actions contains NaN values!")
            logger.warning(f"[REWARD CALC] ❌ Using safe fallback actions")
            current_actions = torch.zeros_like(current_actions)
            
        if torch.isnan(previous_actions).any():
            logger.warning(f"[REWARD CALC] ❌ CRITICAL: previous_actions contains NaN values!")
            logger.warning(f"[REWARD CALC] ❌ Using safe fallback previous actions")
            previous_actions = torch.zeros_like(previous_actions)
        
        # CRITICAL: Validate gate tolerance dictionary
        try:
            for key, value in self.current_gate_tolerance.items():
                if torch.isnan(value).any() if hasattr(value, 'any') else False:
                    logger.warning(f"[REWARD CALC] ❌ CRITICAL: gate_tolerance[{key}] contains NaN values!")
                    logger.warning(f"[REWARD CALC] ❌ Using safe fallback tolerance")
                    # Set safe fallback tolerances
                    self.current_gate_tolerance = {
                        'width': torch.full_like(self.current_gate_tolerance['width'], 1.3),
                        'height_min': torch.full_like(self.current_gate_tolerance['height_min'], 0.5), 
                        'height_max': torch.full_like(self.current_gate_tolerance['height_max'], 2.5)
                    }
                    break
        except Exception as e:
            logger.warning(f"[REWARD CALC] ❌ CRITICAL: Failed to validate gate_tolerance: {e}")
            # Create safe fallback tolerance dictionary
            num_envs = robot_position.shape[0]
            self.current_gate_tolerance = {
                'width': torch.full((num_envs,), 1.3, device=robot_position.device),
                'height_min': torch.full((num_envs,), 0.5, device=robot_position.device),
                'height_max': torch.full((num_envs,), 2.5, device=robot_position.device)
            }
        
        logger.debug(f"[REWARD CALC] ✅ Input validation completed - calling compute_gate_reward")
        
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
            self.current_gate_tolerance,  # Add adaptive tolerance parameter
        )
        
        # CRITICAL: Validate reward outputs
        if torch.isnan(rewards).any():
            logger.warning(f"[REWARD CALC] ❌ CRITICAL: rewards contains NaN values after compute_gate_reward!")
            logger.warning(f"[REWARD CALC] ❌ Setting rewards to zero to prevent neural network crash")
            rewards = torch.zeros_like(rewards)
            
        if torch.isnan(camera_gate_alignment).any():
            logger.warning(f"[REWARD CALC] ❌ CRITICAL: camera_gate_alignment contains NaN values!")
            logger.warning(f"[REWARD CALC] ❌ Setting alignment to zero")
            camera_gate_alignment = torch.zeros_like(camera_gate_alignment)
        
        logger.debug(f"[REWARD CALC] ✅ Reward calculation completed successfully")
        
        # UPDATE EPISODE REWARD TRACKING: Track cumulative reward components
        self.update_episode_reward_tracking(obs_dict, rewards, crashes)
        
        # CRITICAL: Return the computed values (this was missing!)
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
            
            # ===== MULTI-ASPECT CURRICULUM APPLICATION =====
            
            # 1. OBSTACLE COUNT PROGRESSION: Apply new obstacle count behind gate
            obstacles_behind_gate = self.task_config.curriculum.get_obstacle_count_behind_gate(self.curriculum_level)
            
            # CRITICAL FIX: Get the actual number of assets that are always kept from the asset manager
            # instead of hardcoding it as 7. The asset manager determines this based on keep_in_env flags.
            if hasattr(self.sim_env, 'asset_manager') and hasattr(self.sim_env.asset_manager, 'num_keep_in_env'):
                fixed_assets_always_kept = self.sim_env.asset_manager.num_keep_in_env
            else:
                # Fallback: Gate + 6 walls + possibly robot = estimate 8-9, use safe default
                fixed_assets_always_kept = 9  # Updated fallback based on asset manager logs
                
            total_obstacles_in_env = fixed_assets_always_kept + obstacles_behind_gate
            self.obs_dict["num_obstacles_in_env"] = total_obstacles_in_env
            
            # CRITICAL: Also update the environment manager's global tensor dict for asset management
            # This ensures the asset manager gets the updated obstacle count when environments reset
            if hasattr(self.sim_env, 'global_tensor_dict'):
                self.sim_env.global_tensor_dict["num_obstacles_in_env"] = total_obstacles_in_env
            
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
            self.log_curriculum_update(f"   1. OBSTACLES: {obstacles_behind_gate} behind gate (total assets: {total_obstacles_in_env} = {fixed_assets_always_kept} fixed + {obstacles_behind_gate} curriculum)")
            self.log_curriculum_update(f"   2. SPAWN: Using LMF2 config with ±0.5m lateral, ±45° orientation (no curriculum dependency)")
            # Get current randomized angle for first environment (representative)
            current_angle = 0.0
            if hasattr(self, 'static_camera_manager') and hasattr(self.static_camera_manager, 'current_camera_angles'):
                current_angle = self.static_camera_manager.current_camera_angles[0] if self.static_camera_manager.current_camera_angles else 0.0
            self.log_curriculum_update(f"   3. CAMERA ANGLE: ±{self.max_camera_angle:.1f}deg max range, env0: {current_angle:.1f}deg (fixed per episode)")
            
            # ===== GATE SCALING CURRICULUM DEBUG =====
            try:
                from aerial_gym.config.asset_config.gate_scaling_config import GateScalingConfig
                available_scales = GateScalingConfig.get_available_scales_for_level(self.curriculum_level)
                current_avg_scale = torch.mean(self.current_gate_scale).item()
                
                self.log_curriculum_update(f"   4. GATE SCALING: Level {self.curriculum_level} available scales: {available_scales}")
                self.log_curriculum_update(f"      Current average gate scale: {current_avg_scale:.2f} (will change on next resets)")
                
                # Show distribution of current gate scales across environments
                scale_counts = {}
                for scale in [1.0, 0.7, 0.5, 0.4]:
                    count = torch.sum(torch.abs(self.current_gate_scale - scale) < 0.05).item()
                    if count > 0:
                        scale_name = "Full" if scale >= 1.0 else "Medium" if scale >= 0.7 else "Small" if scale >= 0.5 else "Minimum"
                        scale_counts[f"{scale_name}({scale})"] = count
                
                if scale_counts:
                    scale_summary = ", ".join([f"{name}: {count}" for name, count in scale_counts.items()])
                    self.log_curriculum_update(f"      Current gate distribution: {scale_summary}")
                else:
                    self.log_curriculum_update(f"      All gates at default scale 1.0 (expected at level {self.curriculum_level})")
                
                # Show sample tolerance values for the current level
                if available_scales:
                    sample_scale = available_scales[0]  # Show tolerance for first available scale
                    sample_width, sample_height_min, sample_height_max = GateScalingConfig.get_gate_tolerance_for_scale(sample_scale)
                    self.log_curriculum_update(f"      Sample tolerance (scale {sample_scale}): ±{sample_width:.2f}m width, {sample_height_min:.1f}-{sample_height_max:.1f}m height")
                
                self.log_curriculum_update(f"      ⚠️  IMPORTANT: Gate scaling only applies when environments RESET")
                
            except Exception as e:
                self.log_curriculum_update(f"   4. GATE SCALING DEBUG ERROR: {e}")
            
            
            # 5. CAMERA NOISE PROGRESSION (D455 Simulation)
            camera_gaussian_std, camera_dropout_rate = self.task_config.curriculum.get_camera_noise(self.curriculum_level)
            self.log_curriculum_update(f"   5. CAMERA NOISE: Gaussian STD={camera_gaussian_std:.4f}, Dropout={camera_dropout_rate*100:.1f}% (both drone & static)")
            
            # ===== CURRICULUM DEBUGGING: Final state after update =====
            self.log_curriculum_update(f"[CURRICULUM UPDATE] FINAL STATE:")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Level: {self.curriculum_level} (range: {self.task_config.curriculum.min_level}-{self.task_config.curriculum.max_level})")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Max level reached: {self.max_curriculum_level_reached} (NO-DECREASE POLICY)")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Progress: {self.curriculum_progress_fraction:.3f}")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Obstacles behind gate: {obstacles_behind_gate} (total assets: {total_obstacles_in_env} = {fixed_assets_always_kept} fixed + {obstacles_behind_gate} curriculum)")
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
            
            # Add camera angle metrics
            self.infos["curriculum/camera_max_angle"] = torch.tensor(self.max_camera_angle, dtype=torch.float32)
            # Use first environment's angle as representative for wandb tracking
            current_angle = 0.0
            if hasattr(self, 'static_camera_manager') and hasattr(self.static_camera_manager, 'current_camera_angles'):
                current_angle = self.static_camera_manager.current_camera_angles[0] if self.static_camera_manager.current_camera_angles else 0.0
            self.infos["curriculum/camera_current_angle"] = torch.tensor(current_angle, dtype=torch.float32)
            
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
        
        # Gate alignment reward (adaptive tolerance based on gate scale)
        gate_alignment_reward = torch.zeros_like(gate_distance)
        aligned_mask = torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < self.current_gate_tolerance['width']
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
        camera_gate_alignment = torch.clamp(camera_gate_alignment, -1.0, 1.0)  # Clamp to [-1, 1]
        
        # CRITICAL: Replace any NaN values in camera alignment with safe default (no alignment)
        camera_gate_alignment = torch.where(
            torch.isnan(camera_gate_alignment), 
            torch.zeros_like(camera_gate_alignment), 
            camera_gate_alignment
        )
        
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
        
        # Track gate passage rewards (check if any gate passages occurred this step)
        # ADAPTIVE: Use curriculum-based tolerances that scale with gate size
        gate_passed_this_step = (
            (robot_position[:, 1] > self.gate_position[:, 1]) &
            (torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < self.current_gate_tolerance['width']) &
            (robot_position[:, 2] > self.current_gate_tolerance['height_min']) & 
            (robot_position[:, 2] < self.current_gate_tolerance['height_max']) &
            (~self.gate_passed)  # Haven't passed before
        )
        
        # ADAPTIVE: Include both basic passage reward AND center bonus (scales with gate size)
        x_distance_from_center = torch.abs(robot_position[:, 0] - self.gate_position[:, 0])
        z_distance_from_center = torch.abs(robot_position[:, 2] - (self.gate_position[:, 2] + 1.2))
        # Scale center tolerance proportionally with gate scale (50% of width tolerance, 30% of height range)
        center_width_tolerance = self.current_gate_tolerance['width'] * 0.4  # Tighter tolerance for center bonus
        center_height_tolerance = (self.current_gate_tolerance['height_max'] - self.current_gate_tolerance['height_min']) * 0.15
        center_aligned_mask = (x_distance_from_center < center_width_tolerance) & (z_distance_from_center < center_height_tolerance)
        
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
    gate_tolerance,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor], Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    
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
    
    # CRITICAL: Add safeguards for gate distance calculation to prevent NaN propagation
    # Clamp extremely large distances that could cause numerical instability
    gate_distance = torch.clamp(gate_distance, 0.0, 100.0)  # Max distance 100m to prevent overflow
    
    # Replace any NaN values with a safe default distance
    gate_distance = torch.where(torch.isnan(gate_distance), torch.full_like(gate_distance, 10.0), gate_distance)
    
    # Reward for approaching gate
    gate_approach_reward = exponential_reward_function(
        parameter_dict["gate_approach_reward_magnitude"],
        0.5,
        gate_distance,
    )
    
    # Enhanced Camera Facing Reward System - Proportional to alignment angle
    # Calculate vector from drone to gate
    drone_to_gate = gate_position - robot_position
    
    # CRITICAL: Add safeguards for drone_to_gate vector calculation
    # Clamp extremely large vectors that could cause numerical instability
    drone_to_gate = torch.clamp(drone_to_gate, -100.0, 100.0)
    
    # Replace any NaN values with safe default vector pointing forward
    drone_to_gate_safe = torch.where(
        torch.isnan(drone_to_gate), 
        torch.tensor([[1.0, 0.0, 0.0]], device=drone_to_gate.device).expand_as(drone_to_gate), 
        drone_to_gate
    )
    
    drone_to_gate_normalized = drone_to_gate_safe / (torch.norm(drone_to_gate_safe, dim=1, keepdim=True) + 1e-8)
    
    # CRITICAL: Ensure normalized vector is valid
    drone_to_gate_normalized = torch.where(
        torch.isnan(drone_to_gate_normalized),
        torch.tensor([[1.0, 0.0, 0.0]], device=drone_to_gate_normalized.device).expand_as(drone_to_gate_normalized),
        drone_to_gate_normalized
    )
    
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
    
    # CRITICAL: Replace any NaN values in camera alignment with safe default (no alignment)
    camera_gate_alignment = torch.where(
        torch.isnan(camera_gate_alignment), 
        torch.zeros_like(camera_gate_alignment), 
        camera_gate_alignment
    )
    
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
    
    # Reward for gate alignment (being in front of gate opening) - ADAPTIVE TOLERANCE
    gate_alignment_reward = torch.zeros_like(gate_distance)
    # Check if robot is roughly aligned with gate opening (adaptive width tolerance)
    aligned_mask = torch.abs(robot_position[:, 0] - gate_position[:, 0]) < gate_tolerance["width"]
    gate_alignment_reward[aligned_mask] = parameter_dict["gate_alignment_reward_magnitude"]
    
    # Enhanced center alignment rewards for precise gate navigation - ADAPTIVE
    gate_center_bonus = torch.zeros_like(gate_distance)
    # Distance from gate center in X direction (horizontal alignment)
    x_distance_from_center = torch.abs(robot_position[:, 0] - gate_position[:, 0])
    # Distance from gate center in Z direction (vertical alignment)  
    z_distance_from_center = torch.abs(robot_position[:, 2] - (gate_position[:, 2] + 1.2))  # Gate center height ~1.2m
    
    # ADAPTIVE: Scale center tolerance proportionally with gate scale
    center_width_tolerance = gate_tolerance["width"] * 0.4  # Tighter tolerance for center bonus
    center_height_tolerance = (gate_tolerance["height_max"] - gate_tolerance["height_min"]) * 0.15
    center_aligned_mask = (x_distance_from_center < center_width_tolerance) & (z_distance_from_center < center_height_tolerance)
    gate_center_bonus[center_aligned_mask] = parameter_dict["gate_center_bonus_magnitude"]
    
    # Check for gate passage (crossing Y = 0 plane with proper alignment) - ADAPTIVE
    just_passed_gate = (
        (robot_position[:, 1] > gate_position[:, 1]) &  # In front of gate
        (torch.abs(robot_position[:, 0] - gate_position[:, 0]) < gate_tolerance["width"]) &  # Adaptive width tolerance
        (robot_position[:, 2] > gate_tolerance["height_min"]) & 
        (robot_position[:, 2] < gate_tolerance["height_max"]) &  # Adaptive height tolerance
        (~gate_passed)  # Haven't passed before
    )
    
    # Check for center passage (more precise alignment) - ADAPTIVE
    just_passed_center = (
        just_passed_gate &  # Basic passage requirement
        (x_distance_from_center < center_width_tolerance) &  # Adaptive center tolerance
        (z_distance_from_center < center_height_tolerance)    # Adaptive center tolerance
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
