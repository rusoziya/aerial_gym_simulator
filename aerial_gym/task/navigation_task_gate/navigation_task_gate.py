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
from typing import Tuple

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
        
        # If static latents (86:150) are fully ablated, disable static FOV visibility reward
        try:
            import os as _os
            spec_str = _os.environ.get('ABLATE_OBS_RANGES', '').strip()
            static_ablated = False
            if spec_str:
                for spec in [s.strip() for s in spec_str.split(',') if s.strip() and '=' in s]:
                    lhs, rhs = spec.split('=', 1)
                    lhs = lhs.strip(); rhs = rhs.strip()
                    if ':' in lhs:
                        try:
                            a, b = lhs.split(':', 1)
                            a = int(a); b = int(b)
                        except Exception:
                            continue
                        if rhs in ('zero', 'zerograd') and a <= 86 and b >= 150:
                            static_ablated = True
                            break
            if static_ablated:
                try:
                    self.task_config.reward_parameters["static_fov_visibility_reward_magnitude"] = 0.0
                except Exception:
                    pass
        except Exception:
            pass

        # set the each of the elements of reward parameter to a torch tensor
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
        
        # # CONFIG VERIFICATION: Print key reward parameters to verify loading
        # logger.warning("="*60)
        # logger.warning("CONFIG VERIFICATION - REWARD PARAMETERS:")
        # logger.warning(f"pos_reward_magnitude: {self.task_config.reward_parameters['pos_reward_magnitude']}")
        # logger.warning(f"very_close_to_goal_reward_magnitude: {self.task_config.reward_parameters['very_close_to_goal_reward_magnitude']}")
        # logger.warning(f"getting_closer_reward_multiplier: {self.task_config.reward_parameters['getting_closer_reward_multiplier']}")
        # logger.warning(f"gate_approach_reward_magnitude: {self.task_config.reward_parameters['gate_approach_reward_magnitude']}")
        # logger.warning(f"gate_passage_reward_magnitude: {self.task_config.reward_parameters['gate_passage_reward_magnitude']}")
        # logger.warning(f"camera_facing_reward_magnitude: {self.task_config.reward_parameters.get('camera_facing_reward_magnitude', 'NOT FOUND!')}")
        # logger.warning(f"collision_penalty: {self.task_config.reward_parameters['collision_penalty']}")
        # logger.warning("="*60)
        
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
        # Obstacle ablation: if disabled, force to fixed count (default 0)
        try:
            import os
            env_obs_dis = os.getenv('SF_DISABLE_OBSTACLE_RANDOMIZATION', None)
            env_obs_fix = os.getenv('SF_FIXED_OBSTACLES_BEHIND_GATE', None)
            if env_obs_dis is not None:
                obstacles_disable = str(env_obs_dis).lower() == 'true'
            else:
                obstacles_disable = bool(getattr(self.task_config, 'disable_obstacle_randomization', False))
            if env_obs_fix is not None:
                obstacles_fixed = int(env_obs_fix)
            else:
                obstacles_fixed = int(getattr(self.task_config, 'fixed_obstacles_behind_gate', 0))
        except Exception:
            obstacles_disable = bool(getattr(self.task_config, 'disable_obstacle_randomization', False))
            obstacles_fixed = int(getattr(self.task_config, 'fixed_obstacles_behind_gate', 0))
        if obstacles_disable:
            obstacles_behind_gate = max(0, obstacles_fixed)
        else:
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
        
        # Ensure num_envs is available before any observation processing that may rely on it
        self.num_envs = self.sim_env.num_envs
        
        # Propagate ablation flags (gate size and obstacles) to global tensor dict for EnvManager to consume
        try:
            if hasattr(self.sim_env, 'global_tensor_dict'):
                gtd = self.sim_env.global_tensor_dict
                import os
                # Read from env first (worker-safe), fall back to task_config
                env_dis = os.getenv('SF_DISABLE_GATE_SIZE_RANDOMIZATION', None)
                env_fix = os.getenv('SF_FIXED_GATE_SCALE_PERCENT', None)
                if env_dis is not None:
                    disable_flag = str(env_dis).lower() == 'true'
                else:
                    disable_flag = bool(getattr(self.task_config, 'disable_gate_size_randomization', False))
                if env_fix is not None:
                    try:
                        fixed_scale = int(env_fix)
                    except Exception:
                        fixed_scale = int(getattr(self.task_config, 'fixed_gate_scale_percent', 100))
                else:
                    fixed_scale = int(getattr(self.task_config, 'fixed_gate_scale_percent', 100))
                gtd['gate_randomization/disabled'] = disable_flag
                gtd['gate_randomization/fixed_scale_percent'] = fixed_scale
                # Obstacles ablation flags
                if env_obs_dis is not None:
                    obstacles_disable = str(env_obs_dis).lower() == 'true'
                if env_obs_fix is not None:
                    try:
                        obstacles_fixed = int(env_obs_fix)
                    except Exception:
                        obstacles_fixed = int(getattr(self.task_config, 'fixed_obstacles_behind_gate', 0))
                gtd['obstacles_randomization/disabled'] = obstacles_disable
                gtd['obstacles_randomization/fixed_count'] = int(max(0, obstacles_fixed))
                # Static camera orientation randomization ablation flag
                try:
                    env_cam_orient_dis = os.getenv('SF_DISABLE_STATIC_CAMERA_ORIENT_RANDOMIZATION', None)
                    if env_cam_orient_dis is not None:
                        disable_cam_orient = str(env_cam_orient_dis).lower() == 'true'
                    else:
                        disable_cam_orient = bool(getattr(self.task_config, 'disable_static_camera_orientation_randomization', False))
                except Exception:
                    disable_cam_orient = bool(getattr(self.task_config, 'disable_static_camera_orientation_randomization', False))
                gtd['static_camera_randomization/orientation_disabled'] = disable_cam_orient
                # Mirror into the task instance for quick access
                self.disable_static_camera_orientation_randomization = disable_cam_orient
                # Camera noise randomization flags
                try:
                    env_cam_noise_dis = os.getenv('SF_DISABLE_CAMERA_NOISE_RANDOMIZATION', None)
                    if env_cam_noise_dis is not None:
                        disable_cam_noise = str(env_cam_noise_dis).lower() == 'true'
                    else:
                        disable_cam_noise = bool(getattr(self.task_config, 'disable_camera_noise_randomization', False))
                except Exception:
                    disable_cam_noise = bool(getattr(self.task_config, 'disable_camera_noise_randomization', False))
                gtd['camera_randomization/noise_disabled'] = disable_cam_noise
                # Per-camera overrides
                try:
                    env_drone_noise = os.getenv('SF_DISABLE_DRONE_CAMERA_NOISE_RANDOMIZATION', None)
                    env_static_noise = os.getenv('SF_DISABLE_STATIC_CAMERA_NOISE_RANDOMIZATION', None)
                    if env_drone_noise is not None:
                        gtd['camera_randomization/drone_noise_disabled'] = str(env_drone_noise).lower() == 'true'
                    if env_static_noise is not None:
                        gtd['camera_randomization/static_noise_disabled'] = str(env_static_noise).lower() == 'true'
                except Exception:
                    pass
                # Camera frame dropout ablation flag (both drone & static)
                try:
                    env_cam_fd_dis = os.getenv('SF_DISABLE_CAMERA_FRAME_DROPOUT_RANDOMIZATION', None)
                    if env_cam_fd_dis is not None:
                        disable_cam_fd = str(env_cam_fd_dis).lower() == 'true'
                    else:
                        disable_cam_fd = bool(getattr(self.task_config, 'disable_camera_frame_dropout_randomization', False))
                except Exception:
                    disable_cam_fd = bool(getattr(self.task_config, 'disable_camera_frame_dropout_randomization', False))
                gtd['camera_randomization/frame_dropout_disabled'] = disable_cam_fd
                # Per-camera frame-drop overrides
                try:
                    env_drone_fd = os.getenv('SF_DISABLE_DRONE_CAMERA_FRAME_DROPOUT', None)
                    env_static_fd = os.getenv('SF_DISABLE_STATIC_CAMERA_FRAME_DROPOUT', None)
                    if env_drone_fd is not None:
                        gtd['camera_randomization/drone_frame_dropout_disabled'] = str(env_drone_fd).lower() == 'true'
                    if env_static_fd is not None:
                        gtd['camera_randomization/static_frame_dropout_disabled'] = str(env_static_fd).lower() == 'true'
                except Exception:
                    pass
                self.disable_camera_frame_dropout_randomization = disable_cam_fd
                self.disable_camera_noise_randomization = disable_cam_noise
                # Spawn ablation flags (position/orientation): store in global dict
                try:
                    env_spawn_pos_dis = os.getenv('SF_DISABLE_SPAWN_POSITION_RANDOMIZATION', None)
                    env_spawn_orient_dis = os.getenv('SF_DISABLE_SPAWN_ORIENTATION_RANDOMIZATION', None)
                    if env_spawn_pos_dis is not None:
                        disable_spawn_pos = str(env_spawn_pos_dis).lower() == 'true'
                    else:
                        disable_spawn_pos = bool(getattr(self.task_config, 'disable_spawn_position_randomization', False))
                    if env_spawn_orient_dis is not None:
                        disable_spawn_orient = str(env_spawn_orient_dis).lower() == 'true'
                    else:
                        disable_spawn_orient = bool(getattr(self.task_config, 'disable_spawn_orientation_randomization', False))
                except Exception:
                    disable_spawn_pos = bool(getattr(self.task_config, 'disable_spawn_position_randomization', False))
                    disable_spawn_orient = bool(getattr(self.task_config, 'disable_spawn_orientation_randomization', False))
                gtd['spawn_randomization/position_disabled'] = disable_spawn_pos
                gtd['spawn_randomization/orientation_disabled'] = disable_spawn_orient
                # State noise ablation flag (pose)
                try:
                    env_state_noise_dis = os.getenv('SF_DISABLE_STATE_NOISE_RANDOMIZATION', None)
                    if env_state_noise_dis is not None:
                        disable_state_noise = str(env_state_noise_dis).lower() == 'true'
                    else:
                        disable_state_noise = bool(getattr(self.task_config, 'disable_state_noise_randomization', False))
                except Exception:
                    disable_state_noise = bool(getattr(self.task_config, 'disable_state_noise_randomization', False))
                gtd['state_randomization/noise_disabled'] = disable_state_noise
                self.disable_state_noise_randomization = disable_state_noise
                
                # Dynamic camera following ablation flag
                try:
                    env_dynamic_camera_dis = os.environ.get('disable_dynamic_camera_following', None)
                    if env_dynamic_camera_dis is not None:
                        disable_dynamic_camera = str(env_dynamic_camera_dis).lower() == 'true'
                    else:
                        disable_dynamic_camera = bool(getattr(self.task_config, 'disable_dynamic_camera_following', False))
                except Exception:
                    disable_dynamic_camera = bool(getattr(self.task_config, 'disable_dynamic_camera_following', False))
                gtd['dynamic_camera_following/disabled'] = disable_dynamic_camera
                self.disable_dynamic_camera_following = disable_dynamic_camera
                # Static camera yaw sweep controls (curriculum-independent for now)
                try:
                    yaw_sweep_enabled = str(os.getenv('SF_ENABLE_STATIC_CAMERA_YAW_SWEEP', 'false')).lower() == 'true'
                    yaw_sweep_speed_deg = float(os.getenv('SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG', '10.0'))
                except Exception:
                    yaw_sweep_enabled = False
                    yaw_sweep_speed_deg = 10.0
                gtd['static_camera/yaw_sweep_enabled'] = 'true' if yaw_sweep_enabled else 'false'
                gtd['static_camera/yaw_sweep_speed_deg'] = float(yaw_sweep_speed_deg)
                # Locked-follow mode: position fixed, orientation always points at the drone
                try:
                    locked_follow = str(os.getenv('SF_STATIC_CAMERA_LOCKED_FOLLOW', 'false')).lower() in ('1','true','yes','y')
                except Exception:
                    locked_follow = False
                gtd['static_camera/locked_follow'] = locked_follow
                
                # Dynamic camera following enable flag (overrides config setting)
                try:
                    env_dynamic_camera_enable = os.environ.get('enable_dynamic_camera_following', None)
                    if env_dynamic_camera_enable is not None:
                        enable_dynamic_camera = str(env_dynamic_camera_enable).lower() == 'true'
                        # Override the config setting
                        self.task_config.curriculum.enable_dynamic_camera_following = enable_dynamic_camera
                        gtd['dynamic_camera_following/config_overridden'] = True
                        logger.warning(f"[Dynamic Camera] Config overridden by command line: enable_dynamic_camera_following = {enable_dynamic_camera}")
                    else:
                        gtd['dynamic_camera_following/config_overridden'] = False
                except Exception:
                    gtd['dynamic_camera_following/config_overridden'] = False
                
        except Exception:
            pass

        # Immediately select a random gate variant once after creation (safety)
        if hasattr(self.sim_env, 'apply_gate_variant_selection'):
            logger.warning("[GateVariant] Initial selection after build (one-time)")
            self.sim_env.apply_gate_variant_selection(env_ids=torch.arange(self.sim_env.num_envs, device=self.device))
        
        # CRITICAL FIX: Immediately update the environment's obstacle count after creation
        if hasattr(self.sim_env, 'global_tensor_dict'):
            # Override count if obstacle randomization disabled
            try:
                obs_dis = bool(self.sim_env.global_tensor_dict.get('obstacles_randomization/disabled', False))
            except Exception:
                obs_dis = False
            if obs_dis:
                try:
                    fixed_count = int(self.sim_env.global_tensor_dict.get('obstacles_randomization/fixed_count', 0))
                except Exception:
                    fixed_count = 0
                total_obstacles_in_env = fixed_assets_visible + max(0, fixed_count)
            self.sim_env.global_tensor_dict["num_obstacles_in_env"] = total_obstacles_in_env
            logger.info(f"POST-INIT: Updated global_tensor_dict with obstacle count: {total_obstacles_in_env}")

        # Target position: will be set to adaptive gate center on each reset
        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        # REMOVED: target_min_ratio and target_max_ratio sampling logic
        # Target position is now always set to adaptive gate center instead of random sampling

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
        # Force curriculum level if requested via env/task config
        try:
            forced = os.environ.get('SF_FORCE_CURRICULUM_LEVEL', None)
            if forced is None:
                forced = getattr(self.task_config, 'force_curriculum_level', None)
            if forced is not None:
                self.curriculum_level = int(forced)
        except Exception:
            pass
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
            # Determine baseline level and ablation flags
            baseline_level = int(getattr(self.task_config.curriculum, 'min_level', 3))
            pos_dis = False; yaw_dis = False
            try:
                gtd = self.sim_env.global_tensor_dict
                pos_dis = bool(gtd.get('spawn_randomization/position_disabled', False))
                yaw_dis = bool(gtd.get('spawn_randomization/orientation_disabled', False))
            except Exception:
                pass
            # Read spawn ranges for active and baseline
            sr_active = self.task_config.curriculum.get_spawn_ranges(self.curriculum_level)
            sr_base = self.task_config.curriculum.get_spawn_ranges(baseline_level)
            # Select ranges based on ablations
            sr_use = {
                'x_half_span_m': sr_base['x_half_span_m'] if pos_dis else sr_active['x_half_span_m'],
                'y_center_m':    sr_base['y_center_m']    if pos_dis else sr_active['y_center_m'],
                'y_half_span_m': sr_base['y_half_span_m'] if pos_dis else sr_active['y_half_span_m'],
                'z_center_m':    sr_base['z_center_m']    if pos_dis else sr_active['z_center_m'],
                'z_half_span_m': sr_base['z_half_span_m'] if pos_dis else sr_active['z_half_span_m'],
                'yaw_abs_rad':   sr_base['yaw_abs_rad']   if yaw_dis else sr_active['yaw_abs_rad'],
            }
            # Report spawn ablation status in logs
            if pos_dis or yaw_dis:
                status_pos = "DISABLED" if pos_dis else "ENABLED"
                status_yaw = "DISABLED" if yaw_dis else "ENABLED"
                logger.info(f"   2. SPAWN RANDOMIZATION: position={status_pos}, orientation={status_yaw}")
            logger.info(
                f"   2. SPAWN: X∈[{(-sr_use['x_half_span_m']):.1f}, {(+sr_use['x_half_span_m']):.1f}] m, "
                f"Y∈[{(sr_use['y_center_m']-sr_use['y_half_span_m']):.1f}, {(sr_use['y_center_m']+sr_use['y_half_span_m']):.1f}] m, "
                f"Z∈[{(sr_use['z_center_m']-sr_use['z_half_span_m']):.1f}, {(sr_use['z_center_m']+sr_use['z_half_span_m']):.1f}] m; yaw ±{(sr_use['yaw_abs_rad']*57.2958):.1f}°"
            )
        except Exception as e:
            logger.info(f"   2. SPAWN: (fallback) Using fixed LMF2 config due to: {e}")
        # 3. STATIC CAMERA YAW SWEEP STATUS (takes precedence over static orientation randomization)
        try:
            yaw_enabled = str(os.environ.get('SF_ENABLE_STATIC_CAMERA_YAW_SWEEP', 'false')).lower() == 'true'
            yaw_speed = float(os.environ.get('SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG', '10.0'))
        except Exception:
            yaw_enabled = False
            yaw_speed = 10.0
        # Orientation randomization disable flag and dynamic camera effective state
        try:
            cam_orient_disabled = bool(self.sim_env.global_tensor_dict.get('static_camera_randomization/orientation_disabled', False))
        except Exception:
            cam_orient_disabled = False
        try:
            dyn_cfg = getattr(self.task_config.curriculum, 'enable_dynamic_camera_following', False)
            dyn_dis = bool(self.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False))
            dynamic_effective = bool(dyn_cfg and not dyn_dis)
        except Exception:
            dynamic_effective = False
        # Effective note for sweep
        if yaw_enabled:
            if dynamic_effective:
                sweep_note = "IGNORED (dynamic camera active)"
            else:
                sweep_note = "ACTIVE"
        else:
            sweep_note = "N/A"
        logger.info(
            f"   3. STATIC CAMERA YAW SWEEP: {'ENABLED' if yaw_enabled else 'DISABLED'} (speed={yaw_speed:.1f} deg/s) — effective: {sweep_note}; orientation_rand={'DISABLED' if cam_orient_disabled else 'ENABLED'}"
        )
        try:
            base_y = float(os.environ.get('SF_STATIC_CAMERA_BASE_Y', -3.0))
            base_z = float(os.environ.get('SF_STATIC_CAMERA_BASE_Z', 1.5))
        except Exception:
            base_y, base_z = -3.0, 1.5
        logger.info(f"      ↳ static camera base: Y={base_y:.2f} m, Z={base_z:.2f} m")
        # 4. CAMERA ANGLE (randomization applies only when sweep is disabled and dynamic camera is inactive)
        if yaw_enabled and not dynamic_effective:
            logger.info("   4. CAMERA ANGLE: overridden by yaw sweep")
        elif dynamic_effective:
            logger.info("   4. CAMERA ANGLE: suppressed (dynamic camera following active)")
        elif cam_orient_disabled:
            logger.info("   4. CAMERA ANGLE: randomization DISABLED, fixed at 0.0°")
        else:
            logger.info(f"   4. CAMERA ANGLE: ±{self.max_camera_angle:.1f}deg max range (randomized per episode reset, fixed during episode)")
        
        # 5. CAMERA NOISE PROGRESSION (D455 Simulation)
        initial_camera_gaussian_std, initial_camera_dropout_rate = self.task_config.curriculum.get_camera_noise(self.curriculum_level)
        logger.info(f"   5. CAMERA NOISE: Gaussian STD={initial_camera_gaussian_std:.4f}, Dropout={initial_camera_dropout_rate*100:.1f}% (both drone & static)")
        
        # 6. CAMERA FRAME DROPOUT (entire-frame)
        fd = self.task_config.curriculum.get_camera_frame_dropout(self.curriculum_level)
        logger.info(f"   6. CAMERA FRAME DROPOUT: drone_total={fd['drone_total']*100:.1f}% (freeze {fd['drone_freeze']*100:.1f}%, blank {fd['drone_blank']*100:.1f}%), static_total={fd['static_total']*100:.1f}% (freeze {fd['static_freeze']*100:.1f}%, blank {fd['static_blank']*100:.1f}%)")
        
        # 7. STATE NOISE (pose) — new
        state_noise_disabled = False
        try:
            state_noise_disabled = bool(self.sim_env.global_tensor_dict.get('state_randomization/noise_disabled', False))
        except Exception:
            state_noise_disabled = bool(getattr(self, 'disable_state_noise_randomization', False))
        if getattr(self.task_config.curriculum, "enable_state_noise", False) and not state_noise_disabled:
            sn = self.task_config.curriculum.get_state_noise(self.curriculum_level)
            logger.info(
                f"   7. STATE NOISE: drone_pos_std={sn['drone_pos_std_m']:.4f} m, drone_orient_std={sn['drone_orient_std_rad']*57.2958:.3f} deg, "
                f"static_pos_std={sn['static_pos_std_m']:.4f} m, static_orient_std={sn['static_orient_std_rad']*57.2958:.3f} deg"
            )
        else:
            logger.info("   7. STATE NOISE: disabled")
        
        logger.info(f"   8. ASSET MANAGER: Updated both obs_dict and global_tensor_dict with count {total_obstacles_in_env}")
        # Curriculum multiplier debug (initial) - compute fraction directly (attribute may not exist yet)
        try:
            cm_disabled = str(os.environ.get('SF_DISABLE_CURRICULUM_MULTIPLIER', 'false')).lower() == 'true'
        except Exception:
            cm_disabled = False
        if not cm_disabled:
            try:
                cm_disabled = bool(getattr(self.task_config, 'disable_curriculum_multiplier', False))
            except Exception:
                cm_disabled = False
        try:
            frac_current = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)
        except Exception:
            frac_current = 0.0
        frac_eff = 0.0 if cm_disabled else float(frac_current)
        factor = 1.0 + 0.5 * frac_eff
        logger.info(f"   9. CURRICULUM MULTIPLIER: {'DISABLED' if cm_disabled else 'ENABLED'} (factor={factor:.3f})")
        
        # Calculate progress fraction
        self.curriculum_progress_fraction = (
            self.curriculum_level - self.task_config.curriculum.min_level
        ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)
        
        logger.info(f"   8. PROGRESS: {self.curriculum_progress_fraction:.3f} (level {self.curriculum_level}/{self.task_config.curriculum.max_level})")
        logger.info(f"   9. EVALUATION: Check every {self.task_config.curriculum.check_after_log_instances} instances (success rate threshold: {self.task_config.curriculum.success_rate_for_increase:.3f})")
        
        self.log_curriculum_update(f"[INIT] Multi-aspect curriculum initialized at level {self.curriculum_level}")

        # Use dedicated terminations tensor if provided by env_manager; fallback to crashes
        try:
            self.terminations = self.obs_dict["terminations"]
        except Exception:
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
                    shape=(1, 270, 480),
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
        # Episode-level tracking flags
        self._ep_target_success_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
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
        # Static camera FOV visibility reward accumulator (shaped inside-frustum score)
        self.episode_static_fov_visibility_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_boundary_violation_penalty = torch.zeros(self.num_envs, device=self.device)
        
        # Track episode statistics
        self.episode_lengths = torch.zeros(self.num_envs, device=self.device)
        # New: track time-related penalties for analysis
        self.episode_time_penalty = torch.zeros(self.num_envs, device=self.device)
        self.episode_timeout_penalty = torch.zeros(self.num_envs, device=self.device)
        self.completed_episodes = []  # Store last 10 episode breakdowns
        self.max_stored_episodes = 10
        
        # ===== Per-env episode trajectory state (replaces global buffers) =====
        # Mark envs that have just been reset; spawn/last positions captured on first step
        self._episode_fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # Spawn position captured at first step after reset
        self._ep_spawn_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # Gate center captured at spawn for a stable denominator in path efficiency
        self._ep_gate_center_at_spawn = torch.zeros((self.num_envs, 3), device=self.device)
        # Last position for incremental path length accumulation
        self._ep_last_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # Accumulated path length within the episode
        self._ep_path_len = torch.zeros(self.num_envs, device=self.device)
        # Per-env episode step counter (increments each step until reset)
        self._ep_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Min distance to gate center observed within the episode
        self._ep_min_gate_dist = torch.full((self.num_envs,), float('inf'), device=self.device)
        # Crossing state and metrics captured at first crossing
        self._ep_gate_crossed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ep_time_to_gate = torch.full((self.num_envs,), float('nan'), device=self.device)
        self._ep_center_offset_cross = torch.full((self.num_envs,), float('nan'), device=self.device)
        self._ep_height_offset_cross = torch.full((self.num_envs,), float('nan'), device=self.device)
        # Boundary violation debug state for env0 (rising-edge print)
        self._bv_prev_env0 = False
        # Boundary violation per-episode flag (one-shot per env)
        self._bv_flag_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Initialize gate dimensions for all environments after full initialization
        logger.warning("[GATE_ADAPTIVE] Initializing gate dimensions for all environments")
        self.update_gate_dimensions_for_environments(torch.arange(self.sim_env.num_envs, device=self.device))
        
        # Ensure infos survive resets for logging back to the learner
        self._infos_to_return = None

    def logging_sanity_check(self, infos):
        """Sanity check for logging to detect issues with success/crash/timeout logic."""
        successes = infos["successes"]
        crashes = infos["crashes"]
        timeouts = infos["timeouts"]
        time_at_crash = torch.where(
            crashes > 0,
            self.sim_env.sim_steps,
            self.task_config.episode_len_steps * torch.ones_like(self.sim_env.sim_steps),
        )
        env_list_for_toc = (time_at_crash < 5).nonzero(as_tuple=False).squeeze(-1)
        crash_envs = crashes.nonzero(as_tuple=False).squeeze(-1)
        success_envs = successes.nonzero(as_tuple=False).squeeze(-1)
        timeout_envs = timeouts.nonzero(as_tuple=False).squeeze(-1)

        if len(env_list_for_toc) > 0:
            logger.critical("Crash is happening too soon.")
            logger.critical(f"Envs crashing too soon: {env_list_for_toc}")
            logger.critical(f"Time at crash: {time_at_crash[env_list_for_toc]}")

        if torch.sum(torch.logical_and(successes, crashes)) > 0:
            logger.critical("Success and crash are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, successes))}"
            )

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
        # UPDATED: Set target position to adaptive gate center instead of random sampling
        # This ensures "Getting Closer" and "position reward" are consistently aimed at the gate center
        # Gate position comes from obs_dict["env_bounds_min/max"] or is set by environment
        # We need to get gate position from observations after environment reset
        if "gate_position" in self.obs_dict:
            # If gate position is directly available in observations
            self.gate_position[env_ids] = self.obs_dict["gate_position"][env_ids]
        else:
            # Fallback: assume gate is at origin (0, 0, 0) for each environment
            # This matches the typical gate environment setup
            self.gate_position[env_ids, 0] = 0.0  # X: center
            self.gate_position[env_ids, 1] = 0.0  # Y: gate plane
            self.gate_position[env_ids, 2] = 0.0  # Z: ground level
        
        # Robot spawning is now handled by the normal Isaac Gym reset mechanism
        # which uses the min_init_state and max_init_state from LMF2 config
        # This provides ±0.5m lateral variation and ±45° orientation automatically
        
        # Reset gate-specific tracking
        self.gate_passed[env_ids] = False
        # Reset episode flags for these environments
        try:
            self._ep_target_success_flag[env_ids] = False
        except Exception:
            pass
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
        # Propagate eval-stretch flags from task_config if present (for gate size unlocking etc.)
        try:
            stretch_enabled = bool(getattr(self.task_config.curriculum, 'eval_stretch_enabled', False))
            stretch_end = int(getattr(self.task_config.curriculum, 'eval_stretch_end_level', 23))
            self.sim_env.global_tensor_dict['eval_stretch_enabled'] = stretch_enabled
            self.sim_env.global_tensor_dict['eval_stretch_end_level'] = stretch_end
        except Exception:
            pass
        # Update gate dimensions for adaptive rewards after gate selection
        self.update_gate_dimensions_for_environments(env_ids)
        
        # CRITICAL: Set target position to adaptive gate center AFTER gate dimensions are updated
        # This ensures "Getting Closer" and "position reward" terms aim at the correct gate center
        gate_center_x = self.gate_position[env_ids, 0]  # Same X as gate base
        gate_center_y = self.gate_position[env_ids, 1]  # Same Y as gate base  
        gate_center_z = self.gate_position[env_ids, 2] + self.gate_center_height[env_ids]  # Base Z + adaptive center height
        
        self.target_position[env_ids, 0] = gate_center_x
        self.target_position[env_ids, 1] = gate_center_y
        self.target_position[env_ids, 2] = gate_center_z
        
        # Verification logging: show sample target position and gate dimensions for debugging
        if len(env_ids) > 0:
            sample_env = env_ids[0].item()
            sample_target = self.target_position[sample_env]
            sample_gate_pos = self.gate_position[sample_env] 
            sample_center_height = self.gate_center_height[sample_env].item()
            sample_width = self.gate_width[sample_env].item()
            sample_height = self.gate_height[sample_env].item()
            
            logger.debug(f"[TARGET_VERIFICATION] Env {sample_env}: target=({sample_target[0]:.3f}, {sample_target[1]:.3f}, {sample_target[2]:.3f}), gate_base=({sample_gate_pos[0]:.3f}, {sample_gate_pos[1]:.3f}, {sample_gate_pos[2]:.3f}), center_height={sample_center_height:.3f}, dims=({sample_width:.3f}x{sample_height:.3f})")
        
        logger.debug(f"[TARGET_UPDATE] Set target positions for {len(env_ids)} envs to adaptive gate centers")
        
        # Reset per-env episode trajectory state for these environments
        try:
            self._episode_fresh[env_ids] = True
            self._ep_path_len[env_ids] = 0.0
            self._ep_steps[env_ids] = 0
            self._ep_min_gate_dist[env_ids] = float('inf')
            self._ep_gate_crossed[env_ids] = False
            self._ep_time_to_gate[env_ids] = float('nan')
            self._ep_center_offset_cross[env_ids] = float('nan')
            self._ep_height_offset_cross[env_ids] = float('nan')
        except Exception:
            pass
        
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
                    for sub_dir in (
                        "resources/models/environment_assets/gates",
                        "resources/models/environment_assets/smaller gates",
                    ):
                        test_path = os.path.join(base_dir, sub_dir, urdf_filename)
                        if os.path.exists(test_path):
                            urdf_path = test_path
                            break
                    if urdf_path is not None:
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
        # Expose adaptive gate center heights per env to global tensor dict for camera spawning
        try:
            if hasattr(self.sim_env, 'global_tensor_dict'):
                self.sim_env.global_tensor_dict['gate/center_height_per_env'] = self.gate_center_height.detach().clone()
        except Exception:
            pass
    
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
        # NaN/Inf guard: detect invalid actions per env, zero them and mark for truncation
        try:
            invalid_action_mask = torch.any(torch.isnan(transformed_action) | torch.isinf(transformed_action), dim=1)
            if torch.any(invalid_action_mask):
                transformed_action[invalid_action_mask] = 0.0
                # Defer truncation application until after time-limit assignment below
                nan_trunc_mask = invalid_action_mask.clone()
                # Guard debug: log offending envs for invalid actions
                try:
                    if getattr(self.task_config, 'guard_debug_enabled', True):
                        _ids = torch.nonzero(invalid_action_mask, as_tuple=False).squeeze(-1).tolist()
                        logger.warning(f"[NaNGuard] Invalid ACTION in envs {_ids}; zeroed and will truncate")
                except Exception:
                    pass
            else:
                nan_trunc_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        except Exception:
            nan_trunc_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        logger.debug(f"raw_action: {actions[0]}, transformed action: {transformed_action[0]}")
        
        # Pass 4D velocity commands directly to simulation environment
        self.sim_env.step(actions=transformed_action)
        
        # Observation NaN/Inf guard: scan tensors in obs_dict and truncate offending envs
        try:
            invalid_obs_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for k, v in self.obs_dict.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == self.num_envs:
                    bad = torch.any(torch.isnan(v) | torch.isinf(v), dim=tuple(range(1, v.ndim)))
                    # Guard debug: log offending obs keys and envs
                    if getattr(self.task_config, 'guard_debug_enabled', True) and torch.any(bad):
                        try:
                            _ids = torch.nonzero(bad, as_tuple=False).squeeze(-1).tolist()
                            logger.warning(f"[NaNGuard] Invalid OBS '{k}' in envs {_ids}")
                        except Exception:
                            pass
                    invalid_obs_mask |= bad
            if torch.any(invalid_obs_mask):
                # zero-out obvious offenders to keep reward path stable for this step
                try:
                    if "robot_position" in self.obs_dict and isinstance(self.obs_dict["robot_position"], torch.Tensor):
                        rp = self.obs_dict["robot_position"]
                        rp[invalid_obs_mask] = 0.0
                except Exception:
                    pass
                # Guard debug: log envs that will be truncated due to invalid observations
                try:
                    if getattr(self.task_config, 'guard_debug_enabled', True):
                        _ids = torch.nonzero(invalid_obs_mask, as_tuple=False).squeeze(-1).tolist()
                        logger.warning(f"[NaNGuard] Invalid OBS detected; envs {_ids} will truncate")
                except Exception:
                    pass
                nan_trunc_mask |= invalid_obs_mask
        except Exception:
            pass

        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        self.rewards[:], self.terminations[:], camera_gate_alignment = self.compute_rewards_and_crashes(self.obs_dict)
        # Reward NaN/Inf guard: sanitize invalid rewards and truncate offending envs
        try:
            invalid_reward_mask = torch.isnan(self.rewards) | torch.isinf(self.rewards)
            if torch.any(invalid_reward_mask):
                try:
                    if getattr(self.task_config, 'guard_debug_enabled', True):
                        _ids = torch.nonzero(invalid_reward_mask, as_tuple=False).squeeze(-1).tolist()
                        logger.warning(f"[NaNGuard] Invalid REWARD in envs {_ids}; zeroed and truncating.")
                except Exception:
                    pass
                self.rewards[invalid_reward_mask] = 0.0
                # Ensure truncation to reset these envs safely
                self.truncations[invalid_reward_mask] = 1
        except Exception:
            pass

        # logger.info(f"Curricluum Level: {self.curriculum_level}")

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )
        # Apply NaN/Inf-triggered truncations (takes precedence)
        if torch.any(nan_trunc_mask):
            self.truncations[nan_trunc_mask] = 1
            # Guard debug: final truncation set due to NaN/Inf
            try:
                if getattr(self.task_config, 'guard_debug_enabled', True):
                    _ids = torch.nonzero(nan_trunc_mask, as_tuple=False).squeeze(-1).tolist()
                    logger.warning(f"[NaNGuard] Truncating envs due to NaN/Inf: {_ids}")
            except Exception:
                pass

        # ===== SIMPLE GATE PASSAGE SUCCESS CRITERIA =====
        # Success = simply passing through the gate boundary (any part of the gate opening)
        # More forgiving than target-based or centered passage requirements
        robot_position = self.obs_dict["robot_position"]
        # Snapshot positions BEFORE any potential resets mutate the shared tensors
        try:
            robot_position_before_reset = robot_position.clone()
        except Exception:
            robot_position_before_reset = torch.clone(robot_position)
        
        # Gate passage detection: crossed gate plane within the FULL gate opening (100% tolerance)
        # Accept any passage through the opening: width ±50% and height from bottom to top
        gate_success_width_tolerance = self.gate_width * 0.50
        gate_success_min_height = self.gate_position[:, 2]  # gate bottom
        gate_success_max_height = self.gate_position[:, 2] + self.gate_height  # gate top
        
        gate_passage_success = (
            (robot_position[:, 1] > self.gate_position[:, 1]) &  # Crossed gate (Y > 0)
            (torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < gate_success_width_tolerance) &  # Within gate width
            (robot_position[:, 2] > gate_success_min_height) & (robot_position[:, 2] < gate_success_max_height)  # Within gate height range
        )
        
        # Immediate success termination and reset
        # If success occurs now (by gate passage), mark TERMINATION to force reset
        # Target window: within 10% of gate width (X) and 10% of gate height (Z) around ADAPTIVE gate center
        x_off_imm = torch.abs(robot_position[:, 0] - self.gate_position[:, 0])
        z_off_imm = torch.abs(robot_position[:, 2] - (self.gate_position[:, 2] + self.gate_center_height))
        x_ok_imm = x_off_imm <= (self.gate_width * 0.10)
        z_ok_imm = z_off_imm <= (self.gate_height * 0.10)
        target_success_immediate = x_ok_imm & z_ok_imm
        # Do not let target-window success terminate episodes; training success uses gate passage only
        immediate_success_mask = (~(self.terminations > 0)) & gate_passage_success
        if torch.any(immediate_success_mask):
            # Mark terminations immediately so the environment will reset at post_reward_calculation_step
            self.terminations[immediate_success_mask] = 1
            # Record per-episode target success flags where 10% tolerance is also met
            try:
                self._ep_target_success_flag[immediate_success_mask] |= target_success_immediate[immediate_success_mask]
            except Exception:
                pass
            try:
                success_envs = torch.nonzero(immediate_success_mask, as_tuple=False).squeeze(-1).tolist()
            except Exception:
                success_envs = []
            logger.debug(f"[SUCCESS_RESET] Immediate success achieved in envs: {success_envs}. Terminating and resetting.")
        
        # Success when episode TERMINATES (not crashes) and gate passage achieved
        crash_mask = (self.obs_dict["crashes"] > 0)
        successes = (self.terminations > 0) & gate_passage_success & (~crash_mask)
        
        # Target success at truncation: same 10% width/height window around adaptive gate center
        x_off = torch.abs(robot_position[:, 0] - self.gate_position[:, 0])
        z_off = torch.abs(robot_position[:, 2] - (self.gate_position[:, 2] + self.gate_center_height))
        x_ok = x_off <= (self.gate_width * 0.10)
        z_ok = z_off <= (self.gate_height * 0.10)
        target_success = x_ok & z_ok
        target_successes = (self.terminations > 0) & (target_success & gate_passage_success) & (~crash_mask)
        # Also accumulate per-episode target success flag when truncated at step end
        try:
            end_success_mask = (self.terminations > 0) & (target_success & gate_passage_success) & (~crash_mask)
            self._ep_target_success_flag[end_success_mask] = True
        except Exception:
            pass
        
        # Training success remains gate passage only; target success is for logging metrics only
        # successes = torch.logical_or(successes, target_successes)
        

        # ===== END SIMPLE GATE PASSAGE SUCCESS =====
        timeouts = torch.where(
            self.truncations > 0, torch.logical_not(successes), torch.zeros_like(successes)
        )
        timeouts = torch.where(
            self.terminations > 0, torch.zeros_like(timeouts), timeouts
        )  # timeouts are not counted if there is a crash

        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        # Report crashes only (exclude success-based terminations)
        self.infos["crashes"] = self.obs_dict["crashes"]

        # One-off timeout penalty: discourage hover-to-horizon strategies
        try:
            timeout_penalty = float(self.task_config.reward_parameters.get('timeout_penalty', 70.0))
        except Exception:
            timeout_penalty = 75.0
        if torch.any(timeouts):
            # Apply to the per-env reward vector maintained at the task level
            self.rewards = self.rewards - (timeouts.float() * timeout_penalty)
            try:
                self.episode_timeout_penalty[timeouts] -= timeout_penalty
            except Exception:
                pass
        
        # Add gate navigation specific info to wandb tracking
        # Calculate gate navigation metrics from current state
        robot_position = self.obs_dict["robot_position"]
        # Use geometric center of gate opening (z + center_height) so a perfect center pass can approach 0
        gate_center_position = self.gate_position.clone()
        try:
            gate_center_position[:, 2] = gate_center_position[:, 2] + self.gate_center_height
        except Exception:
            pass
        gate_distance = torch.norm(robot_position - gate_center_position, dim=1)
        
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
        self.infos["curriculum/current_level_minus_1"] = torch.tensor(self.curriculum_level - 1, dtype=torch.float32)

        # ===== Per-env episode trajectory state update =====
        # Initialize newly reset envs on their first step
        try:
            if hasattr(self, '_episode_fresh'):
                fresh_mask = self._episode_fresh
                if torch.any(fresh_mask):
                    self._ep_spawn_pos[fresh_mask] = robot_position[fresh_mask]
                    # Store gate CENTER at spawn (z corrected by current center height)
                    try:
                        _gcenter = self.gate_position.clone()
                        _gcenter[:, 2] = _gcenter[:, 2] + self.gate_center_height
                        self._ep_gate_center_at_spawn[fresh_mask] = _gcenter[fresh_mask]
                    except Exception:
                        self._ep_gate_center_at_spawn[fresh_mask] = self.gate_position[fresh_mask]
                    self._ep_last_pos[fresh_mask] = robot_position[fresh_mask]
                    # counters and accumulators already zeroed in reset_idx
                    self._episode_fresh[fresh_mask] = False
            else:
                # Safety init if missing (should not happen)
                self._episode_fresh = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        except Exception:
            pass
        
        # Incremental path length accumulation and min distance tracking
        try:
            step_deltas = robot_position - self._ep_last_pos
            step_dist = torch.norm(step_deltas, dim=1)
            self._ep_path_len += step_dist
            self._ep_last_pos = robot_position
            # Update episode step counters
            self._ep_steps += 1
            # Update min distance to gate center within episode (center-corrected)
            step_gate_dist = torch.norm(robot_position - gate_center_position, dim=1)
            self._ep_min_gate_dist = torch.minimum(self._ep_min_gate_dist, step_gate_dist)
            # Record first crossing time and offsets (2D XZ center error; height error in Z only)
            newly_crossed = (~self._ep_gate_crossed) & gate_passed_current
            if torch.any(newly_crossed):
                self._ep_gate_crossed[newly_crossed] = True
                # Time to gate in episode steps
                self._ep_time_to_gate[newly_crossed] = self._ep_steps[newly_crossed].to(torch.float32)
                # Offsets at crossing relative to ADAPTIVE gate center (2D XZ distance)
                dx_cross = robot_position[newly_crossed, 0] - gate_center_position[newly_crossed, 0]
                dz_cross = robot_position[newly_crossed, 2] - gate_center_position[newly_crossed, 2]
                co = torch.sqrt(dx_cross * dx_cross + dz_cross * dz_cross)
                ho = torch.abs(dz_cross)
                self._ep_center_offset_cross[newly_crossed] = co
                self._ep_height_offset_cross[newly_crossed] = ho
        except Exception:
            pass

        # self.logging_sanity_check(self.infos)  # Disabled per user request
        self.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )
        # rendering happens at the post-reward calculation step since the newer measurement is required to be
        # sent to the RL algorithm as an observation and it helps if the camera image is updated then
        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            # Episode-end trajectory metrics for envs that reset now (computed from per-env state)
            try:
                env_ids = reset_envs if torch.is_tensor(reset_envs) else torch.tensor(reset_envs, device=self.device, dtype=torch.long)
                # Path efficiency = path length / straight-line distance from spawn to gate center at spawn
                denom = torch.norm(self._ep_spawn_pos[env_ids] - self._ep_gate_center_at_spawn[env_ids], dim=1)
                denom = torch.clamp(denom, min=1e-6)
                # Fallback for rare cases where incremental path stayed ~0 (e.g., immediate reset)
                disp = torch.norm((robot_position[env_ids] - self._ep_spawn_pos[env_ids]), dim=1)
                path_len = self._ep_path_len[env_ids]
                path_len = torch.where(path_len <= 1e-6, disp, path_len)
                path_eff = torch.full((self.num_envs,), float('nan'), device=self.device)
                path_eff[env_ids] = (path_len / denom).clamp(max=1000.0)
                # Time to gate in steps (already NaN for non-crossers)
                time_to_gate = self._ep_time_to_gate.clone()
                # Min distance to gate center during episode
                min_gate_dist = self._ep_min_gate_dist.clone()
                # Offsets at crossing (NaN for non-crossers)
                center_offset = self._ep_center_offset_cross.clone()
                height_offset = self._ep_height_offset_cross.clone()
                # Last position at episode end (absolute and center-relative distance)
                # Use the snapshot from BEFORE reset to report end-of-episode last pose
                last_pos = robot_position_before_reset[env_ids]
                last_pos_x = last_pos[:, 0]
                last_pos_y = last_pos[:, 1]
                last_pos_z = last_pos[:, 2]
                # Center-relative error (2D XZ) at termination
                dx_last = last_pos_x - gate_center_position[env_ids, 0]
                dz_last = last_pos_z - gate_center_position[env_ids, 2]
                last_center_offset_vals = torch.sqrt(dx_last * dx_last + dz_last * dz_last)
                last_height_offset_vals = torch.abs(dz_last)
                # Debug print: average across resetting envs (NaN-aware)
                pe_avg = torch.nanmean(path_eff[env_ids])
                ttg_avg = torch.nanmean(time_to_gate[env_ids])
                mgd_avg = torch.nanmean(min_gate_dist[env_ids])
                co_avg = torch.nanmean(center_offset[env_ids])
                ho_avg = torch.nanmean(height_offset[env_ids])
                lpx_avg = torch.nanmean(last_pos_x)
                lpy_avg = torch.nanmean(last_pos_y)
                lpz_avg = torch.nanmean(last_pos_z)
                # Also compute episode-end offsets (useful fallback when no crossing occurred)
                lco_avg = torch.nanmean(last_center_offset_vals)
                lho_avg = torch.nanmean(last_height_offset_vals)
                # Report both: overall success rate (gate passage) and target success rate (gate passage AND 10%/10%)
                try:
                    # Overall success rate among resetting envs
                    overall_success_rate = torch.mean((successes[env_ids] > 0).float())
                    # Target success (10% width/height AND gate passage) among resetting envs
                    target_success_rate = torch.mean((target_successes[env_ids] > 0).float())
                except Exception:
                    overall_success_rate = torch.tensor(float('nan'), device=self.device)
                    target_success_rate = torch.tensor(float('nan'), device=self.device)
                # num = len(env_ids)
                # logger.warning(f"[TRAJ_METRICS][DEBUG][avg over {num} envs] path_efficiency={float(pe_avg.item()):.3f}, time_to_gate_steps={float(ttg_avg.item()):.2f}, min_gate_distance={float(mgd_avg.item()):.3f}, center_offset_success={float(co_avg.item()):.3f}, height_offset_success={float(ho_avg.item()):.3f}, last_position=({float(lpx_avg.item()):.2f},{float(lpy_avg.item()):.2f},{float(lpz_avg.item()):.2f}), last_center_distance={float(lcd_avg.item()):.3f}")
                # Stash per-env episode metrics for worker-side running aggregation
                try:
                    self._last_traj_metrics_per_env = {
                        'path_efficiency': path_eff.detach().clone(),
                        'time_to_gate_steps': time_to_gate.detach().clone(),
                        'min_gate_distance': min_gate_dist.detach().clone(),
                        'center_offset_success': center_offset.detach().clone(),
                        'height_offset_success': height_offset.detach().clone(),
                        'target_success_flag': self._ep_target_success_flag.detach().clone(),
                        'last_position_x': torch.full((self.num_envs,), float('nan'), device=self.device),
                        'last_position_y': torch.full((self.num_envs,), float('nan'), device=self.device),
                        'last_position_z': torch.full((self.num_envs,), float('nan'), device=self.device),
                        'last_center_offset': torch.full((self.num_envs,), float('nan'), device=self.device),
                        'last_height_offset': torch.full((self.num_envs,), float('nan'), device=self.device),
                        'crossed': self._ep_gate_crossed.detach().clone(),
                    }
                    self._last_traj_metrics_per_env['last_position_x'][env_ids] = last_pos_x
                    self._last_traj_metrics_per_env['last_position_y'][env_ids] = last_pos_y
                    self._last_traj_metrics_per_env['last_position_z'][env_ids] = last_pos_z
                    self._last_traj_metrics_per_env['last_center_offset'][env_ids] = last_center_offset_vals
                    self._last_traj_metrics_per_env['last_height_offset'][env_ids] = last_height_offset_vals
                except Exception:
                    self._last_traj_metrics_per_env = None
                # Stash the averaged trajectory metrics for logging
                try:
                    # Fallback to episode-end offsets if crossing-based offsets are NaN
                    try:
                        co_val = co_avg
                        if torch.isnan(co_val):
                            co_val = lco_avg
                    except Exception:
                        co_val = lco_avg
                    try:
                        ho_val = ho_avg
                        if torch.isnan(ho_val):
                            ho_val = lho_avg
                    except Exception:
                        ho_val = lho_avg
                    self._last_traj_metrics_avg = {
                        'path_efficiency': float(pe_avg.item()),
                        'time_to_gate_steps': float(ttg_avg.item()),
                        'time_to_gate': float(ttg_avg.item()),
                        'min_gate_distance': float(mgd_avg.item()),
                        'center_offset_success': float(co_val.item()) if hasattr(co_val, 'item') else float('nan'),
                        'height_offset_success': float(ho_val.item()) if hasattr(ho_val, 'item') else float('nan'),
                        # Duplicate keys to match existing dashboards
                        'center_offset': float(co_val.item()) if hasattr(co_val, 'item') else float('nan'),
                        'height_offset': float(ho_val.item()) if hasattr(ho_val, 'item') else float('nan'),
                        'success_rate': float(overall_success_rate.item()) if hasattr(overall_success_rate, 'item') else float('nan'),
                        'target_success_rate': float(target_success_rate.item()) if hasattr(target_success_rate, 'item') else float('nan'),
                        'last_position_x': float(lpx_avg.item()),
                        'last_position_y': float(lpy_avg.item()),
                        'last_position_z': float(lpz_avg.item()),
                        'last_center_offset': float(lco_avg.item()),
                        'last_height_offset': float(lho_avg.item()),
                    }
                except Exception:
                    self._last_traj_metrics_avg = None
                # Provide averaged metrics to infos['episode_extra_stats'] so learner can push to W&B as a backup
                try:
                    extra = self.infos.get('episode_extra_stats', {})
                    if not isinstance(extra, dict):
                        extra = {}
                    extra.update(self._last_traj_metrics_avg or {})
                    # Expose per-camera noise/frame-drop overrides to W&B, mirroring prior style
                    try:
                        gtd = getattr(self.sim_env, 'global_tensor_dict', {})
                        cam_noise_global = bool(gtd.get('camera_randomization/noise_disabled', False))
                        cam_fd_global = bool(gtd.get('camera_randomization/frame_dropout_disabled', False))
                        drone_noise_dis = bool(gtd.get('camera_randomization/drone_noise_disabled', False)) if 'camera_randomization/drone_noise_disabled' in gtd else cam_noise_global
                        static_noise_dis = bool(gtd.get('camera_randomization/static_noise_disabled', False)) if 'camera_randomization/static_noise_disabled' in gtd else cam_noise_global
                        drone_fd_dis = bool(gtd.get('camera_randomization/drone_frame_dropout_disabled', False)) if 'camera_randomization/drone_frame_dropout_disabled' in gtd else cam_fd_global
                        static_fd_dis = bool(gtd.get('camera_randomization/static_frame_dropout_disabled', False)) if 'camera_randomization/static_frame_dropout_disabled' in gtd else cam_fd_global
                        extra['episode_extra_stats/camera_noise_disabled_drone'] = float(drone_noise_dis)
                        extra['episode_extra_stats/camera_noise_disabled_static'] = float(static_noise_dis)
                        extra['episode_extra_stats/camera_frame_dropout_disabled_drone'] = float(drone_fd_dis)
                        extra['episode_extra_stats/camera_frame_dropout_disabled_static'] = float(static_fd_dis)
                    except Exception:
                        pass
                    self.infos['episode_extra_stats'] = extra
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"Trajectory metrics computation failed: {e}")
            # Stash infos to return to the learner before we clear them in reset
            try:
                self._infos_to_return = dict(self.infos)
            except Exception:
                self._infos_to_return = self.infos
            # Finally, reset environments and mark them fresh for next episode
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
        # DEBUG: Compare per-env drone camera images to ensure diversity
        try:
            if (not hasattr(self, '_drone_cam_debug_last')) or (self.num_task_steps % 200 == 0):
                ne = int(image_obs.shape[0])
                def _mean_env(idx):
                    return float(image_obs[idx].mean().item()) if idx < ne else float('nan')
                def _same(idx):
                    return (idx < ne) and bool(torch.allclose(image_obs[0], image_obs[idx]))
                envs_to_check = [5]  # reduced debug output: only env5
                means = {i: _mean_env(i) for i in envs_to_check}
                sames = {i: _same(i) for i in [1, 5, 8, 12]}
                # logger.warning(
                #     f"[DroneCam] depth shape={tuple(image_obs.shape)} env5_mean={means.get(5, float('nan')):.4f}"
                # )
                self._drone_cam_debug_last = self.num_task_steps
        except Exception:
            pass
        
        # Apply D455 camera noise if enabled and not ablated
        noised_image_obs = image_obs.clone()  # Start with clean image
        camera_noise_disabled = False
        try:
            camera_noise_disabled = bool(self.sim_env.global_tensor_dict.get('camera_randomization/noise_disabled', False))
        except Exception:
            camera_noise_disabled = bool(getattr(self, 'disable_camera_noise_randomization', False))
        # Per-camera override: if set, apply to drone camera processing
        drone_noise_override = bool(self.sim_env.global_tensor_dict.get('camera_randomization/drone_noise_disabled', False)) if hasattr(self.sim_env, 'global_tensor_dict') else False
        if getattr(self.task_config.curriculum, "enable_camera_noise", False):
            # Use current level when enabled; otherwise force minimum schedule (level 3)
            if not camera_noise_disabled and not drone_noise_override:
                gaussian_std, dropout_rate = self.task_config.curriculum.get_camera_noise(self.curriculum_level)
            else:
                gaussian_std, dropout_rate = self.task_config.curriculum.get_camera_noise(3)
            
            # Gaussian noise: add N(0, gaussian_std) to each pixel (depth measurement uncertainty)
            if gaussian_std > 0:
                noise = torch.randn_like(noised_image_obs) * gaussian_std
                noised_image_obs = noised_image_obs + noise
            
            # Pixel dropout: set a fraction of pixels to 1.0 (missing depth readings)
            if dropout_rate > 0:
                dropout_mask = torch.rand_like(noised_image_obs) < dropout_rate
                noised_image_obs = noised_image_obs.masked_fill(dropout_mask, 1.0)  # 1.0 = max depth (no reading)
            
            # Clamp values to valid range [0, 1]
            noised_image_obs = torch.clamp(noised_image_obs, 0.0, 1.0)
        
        # Entire-frame dropout (curriculum-driven)
        frame_dropout_disabled = False
        try:
            frame_dropout_disabled = bool(self.sim_env.global_tensor_dict.get('camera_randomization/frame_dropout_disabled', False))
        except Exception:
            frame_dropout_disabled = bool(getattr(self, 'disable_camera_frame_dropout_randomization', False))
        drone_fd_override = bool(self.sim_env.global_tensor_dict.get('camera_randomization/drone_frame_dropout_disabled', False)) if hasattr(self.sim_env, 'global_tensor_dict') else False
        if getattr(self.task_config.curriculum, "enable_camera_frame_dropout", False):
            # Use current level unless frame-dropout is disabled; noise flag should not affect frame dropout
            if not frame_dropout_disabled and not drone_fd_override:
                fd = self.task_config.curriculum.get_camera_frame_dropout(self.curriculum_level)
            else:
                fd = self.task_config.curriculum.get_camera_frame_dropout(3)
            p_blank = fd.get("drone_blank", 0.0)
            p_freeze = fd.get("drone_freeze", 0.0)
            
            # Ensure buffer exists
            if not hasattr(self, "_prev_drone_depth"):
                self._prev_drone_depth = noised_image_obs.clone()
            
            # Apply frame dropout effects
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
            # DEBUG: Compare per-env drone VAE latents
            try:
                if (not hasattr(self, '_drone_vae_debug_last')) or (self.num_task_steps % 200 == 0):
                    z = self.image_latents
                    ne = int(z.shape[0])
                    def _absmean_env(idx):
                        return float(torch.mean(torch.abs(z[idx])).item()) if idx < ne else float('nan')
                    def _same(idx):
                        return (idx < ne) and bool(torch.allclose(z[0], z[idx]))
                    envs_to_check = [5]  # reduced debug output: only env5
                    means = {i: _absmean_env(i) for i in envs_to_check}
                    sames = {i: _same(i) for i in [1, 5, 8, 12]}
                    # logger.warning(
                    #     f"[DroneVAE] latents shape={tuple(z.shape)} env5_absmean={means.get(5, float('nan')):.4f}"
                    # )
                    self._drone_vae_debug_last = self.num_task_steps
            except Exception:
                pass
    def process_static_camera_observation(self):
        """Process static camera observations with D455 curriculum-dependent noise."""
        try:
            # Request batched capture so each env gets its own image for VAE, while
            # GIF/debug paths will still use env0 via non-batched calls where needed
            static_depth, static_seg = self.static_camera_manager.capture_images(batched=True)
            
            # CRITICAL DEBUG: Log static camera capture success/failure
            if not hasattr(self, '_static_debug_logged'):
                self._static_debug_logged = True
                if static_depth is not None:
                    logger.warning(f"✅ Static camera capture successful: shape={static_depth.shape if hasattr(static_depth, 'shape') else 'N/A'}, type={type(static_depth)}")
                else:
                    logger.warning("❌ Static camera capture failed: static_depth is None")
            # Periodic per-env capture stats to confirm diversity
            try:
                if (not hasattr(self, '_static_cam_debug_last')) or (self.num_task_steps % 200 == 0):
                    if hasattr(static_depth, 'shape') and getattr(static_depth, 'ndim', 0) == 3:
                        x = static_depth  # (N,H,W)
                        ne = int(x.shape[0])
                        def _mean_env(idx):
                            return float(x[idx].mean().item()) if idx < ne else float('nan')
                        def _same(idx):
                            return (idx < ne) and bool(np.allclose(x[0], x[idx])) if isinstance(x, np.ndarray) else bool(torch.allclose(x[0], x[idx]))
                        envs_to_check = [5]  # reduced debug output: only env5
                        means = {i: _mean_env(i) for i in envs_to_check}
                        # sames calculation omitted for brevity
                        # logger.warning(
                        #     f"[StaticCamCapture] depth shape={tuple(x.shape)} env5_mean={means.get(5, float('nan')):.4f}"
                        # )
                        self._static_cam_debug_last = self.num_task_steps
            except Exception:
                pass
            
            if static_depth is not None and self.task_config.vae_config.use_vae:
                # Store clean static camera image (batched) and env0 view for GIF/debug
                if isinstance(static_depth, np.ndarray):
                    static_depth_clean_batched = static_depth.copy()
                    static_depth_clean_env0 = static_depth_clean_batched[0]
                else:
                    static_depth_clean_batched = static_depth.clone()
                    static_depth_clean_env0 = static_depth_clean_batched[0]
                
                # Apply D455 camera noise if enabled and not ablated (operate on batched copy)
                static_depth_noised = static_depth_clean_batched.copy() if isinstance(static_depth_clean_batched, np.ndarray) else static_depth_clean_batched.clone()
                static_noise_override = bool(self.sim_env.global_tensor_dict.get('camera_randomization/static_noise_disabled', False)) if hasattr(self.sim_env, 'global_tensor_dict') else False
                global_noise_disabled = bool(self.sim_env.global_tensor_dict.get('camera_randomization/noise_disabled', False)) if hasattr(self.sim_env, 'global_tensor_dict') else False
                if getattr(self.task_config.curriculum, "enable_camera_noise", False):
                    # Current level unless disabled -> then use level 3 minimum
                    if not global_noise_disabled and not static_noise_override:
                        gaussian_std, dropout_rate = self.task_config.curriculum.get_camera_noise(self.curriculum_level)
                    else:
                        gaussian_std, dropout_rate = self.task_config.curriculum.get_camera_noise(3)
                    
                    # Handle numpy array case
                    if isinstance(static_depth_noised, np.ndarray):
                        if gaussian_std > 0:
                            noise = np.random.normal(0.0, gaussian_std, size=static_depth_noised.shape)
                            static_depth_noised = static_depth_noised + noise
                        if dropout_rate > 0:
                            dropout_mask = np.random.rand(*static_depth_noised.shape) < dropout_rate
                            static_depth_noised[dropout_mask] = 1.0
                        static_depth_noised = np.clip(static_depth_noised, 0.0, 1.0)
                    else:
                        if gaussian_std > 0:
                            noise = torch.randn_like(static_depth_noised) * gaussian_std
                            static_depth_noised = static_depth_noised + noise
                        if dropout_rate > 0:
                            dropout_mask = torch.rand_like(static_depth_noised) < dropout_rate
                            static_depth_noised = static_depth_noised.masked_fill(dropout_mask, 1.0)
                        static_depth_noised = torch.clamp(static_depth_noised, 0.0, 1.0)

                # Entire-frame dropout (curriculum-driven)
                static_fd_override = bool(self.sim_env.global_tensor_dict.get('camera_randomization/static_frame_dropout_disabled', False)) if hasattr(self.sim_env, 'global_tensor_dict') else False
                global_fd_disabled = bool(self.sim_env.global_tensor_dict.get('camera_randomization/frame_dropout_disabled', False)) if hasattr(self.sim_env, 'global_tensor_dict') else False
                if getattr(self.task_config.curriculum, "enable_camera_frame_dropout", False):
                    # Decouple from noise flag: only frame-dropout flags control this schedule
                    fd = self.task_config.curriculum.get_camera_frame_dropout(self.curriculum_level) if (not global_fd_disabled and not static_fd_override) else self.task_config.curriculum.get_camera_frame_dropout(3)
                    p_blank = fd.get("static_blank", 0.0)
                    p_freeze = fd.get("static_freeze", 0.0)
                    # Initialize previous static buffer
                    if not hasattr(self, "_prev_static_depth"):
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
                
                # Store env0-only static camera images for GIF/debug (keep pipeline unchanged)
                self.obs_dict["static_depth_clean"] = static_depth_clean_env0
                # If numpy/tensor batched, select env0 for GIF/debug without altering VAE input
                self.obs_dict["static_depth_noised"] = static_depth_noised[0] if (hasattr(static_depth_noised, 'ndim') and (getattr(static_depth_noised, 'ndim') == 3)) else static_depth_noised
                self.obs_dict["static_seg"] = static_seg
                
                # CRITICAL FIX: Enhanced VAE encoding with detailed debugging
                try:
                    # Convert to tensor and process through VAE (use noised version for training)
                    if isinstance(static_depth_noised, np.ndarray):
                        static_depth_tensor = torch.from_numpy(static_depth_noised).float().to(self.device)
                    else:
                        static_depth_tensor = static_depth_noised

                    # Ensure shape is (num_envs, H, W). If single image (H, W), broadcast to all envs
                    if static_depth_tensor.dim() == 2:
                        static_depth_tensor = static_depth_tensor.unsqueeze(0).expand(self.sim_env.num_envs, -1, -1)
                    elif static_depth_tensor.dim() == 3 and static_depth_tensor.shape[0] != self.sim_env.num_envs:
                        # Safe fallback: pad/trim to num_envs
                        n, h, w = static_depth_tensor.shape
                        if n < self.sim_env.num_envs:
                            reps = (self.sim_env.num_envs + n - 1) // n
                            static_depth_tensor = static_depth_tensor.repeat(reps, 1, 1)[:self.sim_env.num_envs]
                        else:
                            static_depth_tensor = static_depth_tensor[:self.sim_env.num_envs]

                    # Periodic static camera depth summary (match DroneCam style)
                    try:
                        if (not hasattr(self, '_static_cam_depth_logged')) or (self.num_task_steps % 200 == 0):
                            self._static_cam_depth_logged = True
                            depth = static_depth_tensor
                            ne = int(depth.shape[0])
                            def _mean_env(idx):
                                return float(torch.mean(depth[idx]).item()) if idx < ne else float('nan')
                            def _same(idx):
                                return (idx < ne) and bool(torch.allclose(depth[0], depth[idx]))
                            envs_to_check = [0, 1, 5, 8, 12]
                            means = {i: _mean_env(i) for i in envs_to_check}
                            sames = {i: _same(i) for i in [1, 5, 8, 12]}
                            # logger.warning(
                            #     f"[StaticCam] depth shape={tuple(depth.shape)} "
                            #     f"env0_mean={means[0]:.4f} env1_mean={means.get(1, float('nan')):.4f} "
                            #     f"env5_mean={means.get(5, float('nan')):.4f} env8_mean={means.get(8, float('nan')):.4f} "
                            #     f"env12_mean={means.get(12, float('nan')):.4f} "
                            #     f"same0_1={sames.get(1, False)} same0_5={sames.get(5, False)} "
                            #     f"same0_8={sames.get(8, False)} same0_12={sames.get(12, False)}"
                            # )
                    except Exception:
                        pass

                    # CRITICAL DEBUG: Log VAE encoding attempt (once)
                    if not hasattr(self, '_vae_debug_logged'):
                        self._vae_debug_logged = True
                        logger.warning(f"🔧 VAE encoding static camera: input_shape={static_depth_tensor.shape}, device={static_depth_tensor.device}")

                    encoded_latents = self.shared_vae_model.encode(static_depth_tensor)
                    self.static_image_latents[:] = encoded_latents

                    # CRITICAL DEBUG: Verify VAE output periodically and compare across envs
                    try:
                        if (not hasattr(self, '_vae_output_logged')) or (self.num_task_steps % 200 == 0):
                            self._vae_output_logged = True
                            z = encoded_latents
                            ne = int(z.shape[0])
                            def _absmean_env(idx):
                                return float(torch.mean(torch.abs(z[idx])).item()) if idx < ne else float('nan')
                            def _same(idx):
                                return (idx < ne) and bool(torch.allclose(z[0], z[idx]))
                            envs_to_check = [5]  # reduced debug output: only env5
                            means = {i: _absmean_env(i) for i in envs_to_check}
                            # logger.warning(
                            #     f"[StaticCamVAE] latents shape={tuple(z.shape)} env5_absmean={means.get(5, float('nan')):.4f}"
                            # )
                    except Exception:
                        pass
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
        image_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
        image_obs[image_obs < 0] = 10.0
        self.min_pixel_dist = torch.amin(image_obs, dim=(1, 2))
        
        # Calculate image rewards for debugging
        mag = self.task_config.reward_parameters.get("image_penalty_magnitude", 4.0)
        expo = self.task_config.reward_parameters.get("image_penalty_exponent", 1.0)
        image_rewards = -exponential_reward_function(
            float(mag), float(expo), self.min_pixel_dist[~self.terminations]
        )
        # Apply image penalty to non-terminated environments
        self.rewards[~self.terminations] += image_rewards
        
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
        self.rewards[~self.terminations] += image_rewards

    def get_return_tuple(self):
        self.process_obs_for_task()
        # If we have stashed infos from the previous step (pre-reset), use them once
        if hasattr(self, '_infos_to_return') and self._infos_to_return is not None:
            infos_to_return = self._infos_to_return
            self._infos_to_return = None
        else:
            infos_to_return = self.infos
        
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
        # Periodic static camera obs debug (env 0): print every 120 steps
        try:
            if hasattr(self, 'num_task_steps') and (int(self.num_task_steps) % 120 == 0):
                if hasattr(self, 'task_obs') and 'observations' in self.task_obs:
                    obs_all = self.task_obs['observations']
                    if obs_all.shape[0] > 0 and obs_all.shape[1] >= 9:
                        sp = obs_all[0, 3:6]
                        so = obs_all[0, 6:9]
                        # Drone and camera world positions
                        rp = self.obs_dict.get('robot_position', None)
                        rp0 = rp[0] if isinstance(rp, torch.Tensor) and rp.shape[0] > 0 else torch.zeros(3, device=self.device)
                        cw = getattr(self, '_debug_cam_world', None)
                        cw0 = cw[0] if isinstance(cw, torch.Tensor) and cw.shape[0] > 0 else torch.zeros(3, device=self.device)
                        # Drone & camera world orientations (Euler)
                        de = getattr(self, '_debug_drone_eul', None)
                        ce = getattr(self, '_debug_cam_eul', None)
                        de0 = de[0] if isinstance(de, torch.Tensor) and de.shape[0] > 0 else torch.zeros(3, device=self.device)
                        ce0 = ce[0] if isinstance(ce, torch.Tensor) and ce.shape[0] > 0 else torch.zeros(3, device=self.device)
                        # logger.warning(
                        #     f"[StaticObs] step={int(self.num_task_steps)} env0 "
                        #     f"drone_w=({rp0[0].item():+.3f},{rp0[1].item():+.3f},{rp0[2].item():+.3f}) "
                        #     f"cam_w=({cw0[0].item():+.3f},{cw0[1].item():+.3f},{cw0[2].item():+.3f}) "
                        #     f"drone_eul_w=({de0[0].item():+.3f},{de0[1].item():+.3f},{de0[2].item():+.3f}) "
                        #     f"cam_eul_w=({ce0[0].item():+.3f},{ce0[1].item():+.3f},{ce0[2].item():+.3f}) "
                        #     f"pos_rel=({sp[0].item():+.3f},{sp[1].item():+.3f},{sp[2].item():+.3f}) "
                        #     f"eul_rel=({so[0].item():+.3f},{so[1].item():+.3f},{so[2].item():+.3f})"
                        # )
        except Exception:
            pass
        
        # Update dynamic camera following every frame if enabled and not disabled by flag
        dynamic_enabled = getattr(self.task_config.curriculum, 'enable_dynamic_camera_following', False)
        dynamic_disabled = bool(self.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False))
        
        if dynamic_enabled and not dynamic_disabled:
            self.static_camera_manager.update_dynamic_camera_following(
                self.obs_dict["robot_position"], 
                self.gate_position, 
                self.gate_center_height
            )
        
        # If static camera yaw sweep is enabled, update static camera orientation every frame
        try:
            gtd = self.sim_env.global_tensor_dict
            sweep_enabled_flag = str(gtd.get('static_camera/yaw_sweep_enabled', 'false')).lower() == 'true'
            # Do not run sweep updates when dynamic following is active
            locked_follow = bool(gtd.get('static_camera/locked_follow', False))
            if locked_follow and not (dynamic_enabled and not dynamic_disabled):
                # Update orientation-only to keep the drone centered
                self.static_camera_manager.update_locked_follow(self.obs_dict["robot_position"]) 
            elif sweep_enabled_flag and not (dynamic_enabled and not dynamic_disabled):
                env_ids_all = torch.arange(self.sim_env.num_envs, device=self.device)
                self.static_camera_manager.update_camera_positions(self.curriculum_level, env_ids_all)
                # Temporary debug for yaw sweep status (print ~once every 5 seconds at 60Hz)
                try:
                    sim_steps = int(gtd.get('sim_steps', torch.tensor([0], device=self.device))[0].item())
                except Exception:
                    sim_steps = 0
                # Reduce spam: every 300 steps instead of every 60
                if (sim_steps % 300) == 0:
                    logger.warning("[YawSweep] Camera orientation update running (sweeping/locked-follow active)")
        except Exception as e:
            logger.debug(f"[YawSweep] Per-step update skipped due to: {e}")
        
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            infos_to_return,
        )

    def _get_static_camera_pose_relative_to_drone(self):
        """Compute per-environment static camera pose and orientation relative to the drone.

        Position: camera_world - robot_world, rotated into drone/body frame (obs[3:6]).
        Orientation: Euler XYZ of q_rel = q_drone^-1 ⊗ q_cam (obs[6:9]).
        Camera world pose reflects either static base placement (with yaw sweep/randomization)
        or dynamic following when enabled.
        """
        device = self.device
        num_envs = getattr(self, "num_envs", 1)

        # Base Y from task_config first, then env var, else default
        try:
            base_y = float(getattr(self.task_config, 'static_camera_base_y', float(os.environ.get('SF_STATIC_CAMERA_BASE_Y', -3.0))))
        except Exception:
            base_y = -3.0

        # Base Z can be numeric or 'adaptive'
        adaptive_z = False
        base_z_value = 1.5
        try:
            cfg_base_z = getattr(self.task_config, 'static_camera_base_z', None)
            if cfg_base_z is not None:
                if isinstance(cfg_base_z, str) and cfg_base_z.strip().lower() == 'adaptive':
                    adaptive_z = True
                else:
                    base_z_value = float(cfg_base_z)
            else:
                env_base_z = os.environ.get('SF_STATIC_CAMERA_BASE_Z', None)
                if env_base_z is not None and str(env_base_z).strip().lower() == 'adaptive':
                    adaptive_z = True
                elif env_base_z is not None:
                    base_z_value = float(env_base_z)
        except Exception:
            pass

        # Resolve per-env Z when adaptive
        if adaptive_z:
            try:
                gate_center_z = getattr(self, 'gate_center_height', None)
                if gate_center_z is None:
                    gate_center_z = torch.full((num_envs,), 1.5, device=device, dtype=torch.float32)
                elif not torch.is_tensor(gate_center_z):
                    gate_center_z = torch.full((num_envs,), float(gate_center_z), device=device, dtype=torch.float32)
                else:
                    gate_center_z = gate_center_z.to(device=device, dtype=torch.float32).view(-1)
            except Exception:
                gate_center_z = torch.full((num_envs,), 1.5, device=device, dtype=torch.float32)
        else:
            gate_center_z = torch.full((num_envs,), float(base_z_value), device=device, dtype=torch.float32)

        # Determine if dynamic camera following is effective (enabled and not disabled)
        dynamic_enabled = bool(getattr(self.task_config.curriculum, 'enable_dynamic_camera_following', False))
        try:
            dyn_dis = bool(self.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False)) if hasattr(self, 'sim_env') and hasattr(self.sim_env, 'global_tensor_dict') else False
        except Exception:
            dyn_dis = False
        dynamic_effective = bool(dynamic_enabled and not dyn_dis)

        # Camera world positions for each env
        cam_world = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        if dynamic_effective:
            # Follow the drone with fixed offset
            try:
                from aerial_gym.config.task_config.navigation_task_config_gate import task_config as _tc
                x_off, y_off, z_off = _tc.curriculum.get_dynamic_camera_follow_offset()
            except Exception:
                x_off, y_off, z_off = 0.0, -1.0, 0.0
            try:
                robot_pos_world = self.obs_dict['robot_position'].to(device=device, dtype=torch.float32)
            except Exception:
                robot_pos_world = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
            cam_world[:, 0] = robot_pos_world[:, 0] + float(x_off)
            cam_world[:, 1] = robot_pos_world[:, 1] + float(y_off)
            cam_world[:, 2] = robot_pos_world[:, 2] + float(z_off)
        else:
            # Static base placement (x=0, y=base_y, z=gate_center or fixed)
            cam_world[:, 0] = 0.0
            cam_world[:, 1] = float(base_y)
            cam_world[:, 2] = gate_center_z

        # Robot world pose tensors
        try:
            robot_pos = self.obs_dict['robot_position'].to(device=device, dtype=torch.float32)
        except Exception:
            robot_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        # Prefer vehicle (yaw-only) orientation for stable horizontal frame; fallback to body
        q = None
        try:
            q = self.obs_dict.get('robot_vehicle_orientation', None)
        except Exception:
            q = None
        if q is None:
            try:
                q = self.obs_dict.get('robot_orientation', None)
            except Exception:
                q = None
        if q is None:
            q = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
            q[:, 3] = 1.0
        else:
            q = q.to(device=device, dtype=torch.float32)

        # Relative position in world then rotate into the drone/body frame
        rel_world = cam_world - robot_pos
        rel_pos_body = quat_rotate_inverse(q, rel_world)

        # Compute camera world orientation as look-at towards a per-env target
        # Target is the adaptive gate center when static base or dynamic following
        try:
            gate_pos_world = self.gate_position
        except Exception:
            gate_pos_world = torch.zeros((num_envs, 3), device=device)
        try:
            gcz = self.gate_center_height
        except Exception:
            gcz = torch.full((num_envs,), 1.5, device=device)
        target_world = gate_pos_world.clone()
        target_world[:, 2] = gate_pos_world[:, 2] + gcz

        # When static base+ yaw sweep/randomization is active, adjust target X/Y using current yaw offset if available
        if not dynamic_effective:
            try:
                scm = getattr(self, 'static_camera_manager', None)
                have_angles = (scm is not None) and hasattr(scm, 'current_camera_angles') and (len(scm.current_camera_angles) >= num_envs)
            except Exception:
                have_angles = False
            if have_angles:
                # Build per-env target consistent with update_camera_positions
                target_distance = torch.abs(cam_world[:, 1])  # |base_y|
                angles_deg = torch.tensor(scm.current_camera_angles[:num_envs], dtype=torch.float32, device=device)
                ang = angles_deg * (3.141592653589793 / 180.0)
                target_world[:, 0] = cam_world[:, 0] + target_distance * torch.sin(ang)
                target_world[:, 1] = cam_world[:, 1] + target_distance * torch.cos(ang)

        # Camera forward vector and Euler in world
        fwd = target_world - cam_world
        fwd = fwd / (torch.norm(fwd, dim=1, keepdim=True) + 1e-8)
        # Yaw: angle in XY toward +Y; Pitch: elevation toward +Z; Roll=0 (world up maintained)
        fx, fy, fz = fwd[:, 0], fwd[:, 1], fwd[:, 2]
        yaw_cam = torch.atan2(fx, torch.clamp(fy, min=1e-8))
        hyp = torch.sqrt(torch.clamp(fx * fx + fy * fy, min=1e-8))
        pitch_cam = torch.atan2(fz, hyp)
        roll_cam = torch.zeros_like(yaw_cam)
        eul_cam = torch.stack([roll_cam, pitch_cam, yaw_cam], dim=1)
        q_cam = quat_from_euler_xyz_tensor(eul_cam)

        # Relative orientation q_rel = q_drone^-1 ⊗ q_cam; then Euler in drone/body frame
        q_drone = q
        q_drone_conj = torch.stack([-q_drone[:, 0], -q_drone[:, 1], -q_drone[:, 2], q_drone[:, 3]], dim=1)
        q_rel = quat_mul(q_drone_conj, q_cam)
        rel_orient_euler = ssa(get_euler_xyz_tensor(q_rel))
        drone_eul_world = ssa(get_euler_xyz_tensor(q_drone))
        # Stash debug state for printing after obs assembly
        try:
            self._debug_cam_world = cam_world.detach().clone()
            self._debug_rel_pos = rel_pos_body.detach().clone()
            self._debug_rel_eul = rel_orient_euler.detach().clone()
            self._debug_cam_eul = eul_cam.detach().clone()
            self._debug_drone_eul = drone_eul_world.detach().clone()
        except Exception:
            pass
        return rel_pos_body, rel_orient_euler

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
        if getattr(self.task_config.curriculum, "enable_state_noise", False) and not bool(self.sim_env.global_tensor_dict.get('state_randomization/noise_disabled', False)):
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
        if getattr(self.task_config.curriculum, "enable_state_noise", False) and not bool(self.sim_env.global_tensor_dict.get('state_randomization/noise_disabled', False)):
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
        if getattr(self.task_config.curriculum, "enable_state_noise", False) and not bool(self.sim_env.global_tensor_dict.get('state_randomization/noise_disabled', False)):
            noise_cfg = self.task_config.curriculum.get_state_noise(self.curriculum_level)
            do_std = float(noise_cfg.get("drone_orient_std_rad", 0.0))
            if do_std > 0.0:
                euler_angles = euler_angles + torch.randn_like(euler_angles) * do_std
                euler_angles = torch.atan2(torch.sin(euler_angles), torch.cos(euler_angles))
        self.task_obs["observations"][:, 9:12] = euler_angles  # MODIFIED: Include full yaw instead of setting to 0.0

        # ===== DRONE STATE (VELOCITIES) AND ACTIONS =====
        # [12:15] = Robot body linear velocity (3D)
        try:
            self.task_obs["observations"][:, 12:15] = self.obs_dict["robot_body_linvel"]
        except Exception:
            pass

        # [15:18] = Robot body angular velocity (3D)
        try:
            self.task_obs["observations"][:, 15:18] = self.obs_dict["robot_body_angvel"]
        except Exception:
            pass

        # [18:22] = Last applied robot actions (4D for gate navigation)
        try:
            self.task_obs["observations"][:, 18:22] = self.obs_dict["robot_actions"]
        except Exception:
            pass

        # ===== CAMERA LATENTS (DRONE AND STATIC, 64D EACH) =====
        # [22:86] = Drone camera VAE latents (64D)
        try:
            if hasattr(self, "image_latents") and isinstance(self.image_latents, torch.Tensor):
                if self.image_latents.shape[1] >= 64:
                    self.task_obs["observations"][:, 22:86] = self.image_latents[:, :64]
        except Exception:
            pass

        # [86:150] = Static camera VAE latents (64D)
        try:
            if hasattr(self, "static_image_latents") and isinstance(self.static_image_latents, torch.Tensor):
                if self.static_image_latents.shape[1] >= 64:
                    self.task_obs["observations"][:, 86:150] = self.static_image_latents[:, :64]
        except Exception:
            pass

        # (Removed W&B latent stats logging per request)

        # Final observation NaN/Inf guard: sanitize outgoing observations tensor
        try:
            obs_tensor = self.task_obs.get("observations", None)
            if isinstance(obs_tensor, torch.Tensor):
                bad = torch.isnan(obs_tensor) | torch.isinf(obs_tensor)
                if torch.any(bad):
                    if getattr(self.task_config, 'guard_debug_enabled', True):
                        logger.warning(f"[NaNGuard] Sanitizing {int(torch.sum(bad).item())} invalid obs entries before return.")
                    obs_tensor[bad] = 0.0
        except Exception:
            pass

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
        
        # First-step stabilization: align previous error/actions with current for fresh episodes
        prev_actions_for_reward = previous_actions
        try:
            fresh_mask = getattr(self, '_episode_fresh', None)
        except Exception:
            fresh_mask = None
        if isinstance(fresh_mask, torch.Tensor) and fresh_mask.shape[0] == self.num_envs:
            if torch.any(fresh_mask):
                # Set previous error equal to current on the first step after reset
                self.pos_error_vehicle_frame_prev[fresh_mask] = self.pos_error_vehicle_frame[fresh_mask]
                # Also zero action diff on the first step after reset
                prev_actions_for_reward = previous_actions.clone()
                prev_actions_for_reward[fresh_mask] = current_actions[fresh_mask]
        
        # Curriculum multiplier ablation: pass effective fraction to scripted reward
        try:
            cm_disabled = str(os.environ.get('SF_DISABLE_CURRICULUM_MULTIPLIER', 'false')).lower() == 'true'
        except Exception:
            cm_disabled = False
        if not cm_disabled:
            try:
                cm_disabled = bool(getattr(self.task_config, 'disable_curriculum_multiplier', False))
            except Exception:
                cm_disabled = False
        try:
            frac_current = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)
        except Exception:
            frac_current = 0.0
        frac_eff = 0.0 if cm_disabled else float(frac_current)
        self._curriculum_multiplier_factor = 1.0 + 0.5 * frac_eff

        # Compute one-shot boundary violation mask in Python: trigger when crossing to front side
        # outside passage window and not yet passed in this episode; apply once and latch
        try:
            y_margin = 0.2
            behind_gate_mask = (robot_position[:, 1] > (self.gate_position[:, 1] + y_margin))
            gate_passage_width_tolerance = self.gate_width * 0.5
            gate_min_height = self.gate_position[:, 2] + self.gate_height * 0.0
            gate_max_height = self.gate_position[:, 2] + self.gate_height * 1.0
            within_passage_window = (
                (torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < gate_passage_width_tolerance)
                & (robot_position[:, 2] > gate_min_height)
                & (robot_position[:, 2] < gate_max_height)
            )
            misaligned_cross_mask = behind_gate_mask & (~within_passage_window) & (~self.gate_passed)
            # Latch: apply penalty only once per episode per env
            if not hasattr(self, '_bv_flag_episode'):
                self._bv_flag_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            boundary_violation_one_shot_mask = misaligned_cross_mask & (~self._bv_flag_episode)
            # Update latch for envs that just violated
            self._bv_flag_episode |= boundary_violation_one_shot_mask
            # Debug detection: log when boundary violation triggers, including where it happened
            if getattr(self.task_config, 'guard_debug_enabled', True) and torch.any(boundary_violation_one_shot_mask):
                try:
                    _ids = torch.nonzero(boundary_violation_one_shot_mask, as_tuple=False).squeeze(-1).tolist()
                    logger.warning(f"[BoundaryViolation] One-shot penalty applied in envs {_ids}")
                    # Print per-env details (limit to first few to avoid spam)
                    try:
                        max_list = int(getattr(self.task_config, 'reward_outlier_log_limit_per_step', 8))
                    except Exception:
                        max_list = 8
                    for eid in _ids[:max_list]:
                        try:
                            rx = float(self.obs_dict["robot_position"][eid, 0].item())
                            ry = float(self.obs_dict["robot_position"][eid, 1].item())
                            rz = float(self.obs_dict["robot_position"][eid, 2].item())
                            gx = float(self.gate_position[eid, 0].item())
                            gy = float(self.gate_position[eid, 1].item())
                            gz = float(self.gate_position[eid, 2].item())
                            gw = float(self.gate_width[eid].item() if hasattr(self.gate_width, 'shape') else self.gate_width)
                            gh = float(self.gate_height[eid].item() if hasattr(self.gate_height, 'shape') else self.gate_height)
                            tol = float((self.gate_width[eid] * 0.5).item()) if hasattr(self.gate_width, 'shape') else float(self.gate_width * 0.5)
                            zmin = float(gz + 0.0 * gh)
                            zmax = float(gz + 1.0 * gh)
                            x_off = abs(rx - gx)
                            logger.warning(
                                f"[BoundaryViolation] Env{eid} pos=({rx:.3f},{ry:.3f},{rz:.3f}), gate_y={gy:.3f}, x_off={x_off:.3f} (tol={tol:.3f}), z_window=({zmin:.3f},{zmax:.3f}), gate_size=(w={gw:.3f}, h={gh:.3f})"
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            boundary_violation_one_shot_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Disable nvFuser optimizations for this scripted block to avoid CUDA fuser fallback warnings
        # (no change in semantics; prevents noisy warnings from TorchScript fuser)
        with torch.jit.optimized_execution(False):
            rewards, crashes, camera_gate_alignment = compute_gate_reward(
                self.pos_error_vehicle_frame,
                self.pos_error_vehicle_frame_prev,
                obs_dict["crashes"],
                current_actions,
                prev_actions_for_reward,
                robot_position,
                robot_vehicle_orientation,
                self.gate_position,
                self.gate_passed,
                frac_eff,
                self.task_config.reward_parameters,
                self.gate_width,
                self.gate_height,
                self.gate_center_height,
                boundary_violation_one_shot_mask,
            )

        # Per-step time cost (scaled like other dense shaping)
        # r_time = -lambda0 * (1 + lambda1 * s^p),  s = step / horizon
        try:
            rp = getattr(self.task_config, 'reward_parameters', {})
        except Exception:
            rp = {}
        try:
            H = int(getattr(self.task_config, 'episode_len_steps', 100))
            if H <= 0:
                H = 100
        except Exception:
            H = 100
        lam1 = float(rp.get('time_penalty_lambda1', 1.0))
        p = float(rp.get('time_penalty_exponent', 2.0))
        desired_total = float(rp.get('time_penalty_total_at_horizon', 4.0))
        lam0 = rp.get('time_penalty_lambda0', None)
        if lam0 is None:
            if abs(p - 2.0) < 1e-6:
                sum_s_p = ((H + 1.0) * (2.0 * H + 1.0)) / (6.0 * H)
            else:
                sum_s_p = H / (p + 1.0)
            denom = H + lam1 * sum_s_p
            lam0 = desired_total / max(denom, 1e-6)
        lam0 = float(lam0)
        try:
            s = torch.clamp((self.episode_lengths + 1.0) / float(H), 0.0, 1.0)
        except Exception:
            s = torch.full((self.num_envs,), 1.0 / float(H), device=self.device)
        raw_time_penalty = -lam0 * (1.0 + lam1 * torch.pow(s, p))
        reward_scale = float(rp.get('reward_scale', 0.1))
        try:
            mult_factor = float(getattr(self, '_curriculum_multiplier_factor', 1.0))
        except Exception:
            mult_factor = 1.0
        time_penalty = (mult_factor * reward_scale * raw_time_penalty)
        rewards = rewards + time_penalty
        # Accumulate for per-episode debugging
        try:
            non_terminated = ~self.terminations
            if torch.sum(non_terminated) > 0:
                self.episode_time_penalty[non_terminated] += time_penalty[non_terminated]
        except Exception:
            pass

        # Static camera FOV visibility reward (depth-based frustum check, shaped)
        try:
            try:
                fov_mag = float(self.task_config.reward_parameters.get("static_fov_visibility_reward_magnitude", 0.0))
            except Exception:
                fov_mag = 0.0
            # If env var SF_ENABLE_STATIC_FOV_REWARD is not explicitly true, force-disable
            try:
                _env_flag = os.environ.get('SF_ENABLE_STATIC_FOV_REWARD', '').strip().lower()
                if _env_flag not in ('1', 'true', 'yes', 'y'):
                    fov_mag = 0.0
            except Exception:
                fov_mag = 0.0
            try:
                fov_alpha = float(self.task_config.reward_parameters.get("static_fov_visibility_exponent", 2.0))
            except Exception:
                fov_alpha = 2.0
            if fov_mag != 0.0:
                # Camera base position (x=0, y=base_y, z either adaptive gate center or fixed 1.5)
                try:
                    parent = getattr(self, 'sim_env', None)
                    gtd = parent.global_tensor_dict if (parent is not None and hasattr(parent, 'global_tensor_dict')) else {}
                except Exception:
                    gtd = {}
                try:
                    base_y = float(os.environ.get('SF_STATIC_CAMERA_BASE_Y', gtd.get('static_camera/base_y', -3.0)))
                except Exception:
                    base_y = -3.0
                try:
                    base_z_env = os.environ.get('SF_STATIC_CAMERA_BASE_Z', None)
                    if base_z_env is None:
                        base_z_env = gtd.get('static_camera/base_z', 1.5)
                    adaptive_z = isinstance(base_z_env, str) and base_z_env.strip().lower() == 'adaptive'
                except Exception:
                    adaptive_z = False

                # Resolve Z per env
                if adaptive_z:
                    gate_center_z = self.gate_center_height
                else:
                    gate_center_z = torch.full((self.num_envs,), 1.5, device=self.device)

                cam_pos = torch.stack([
                    torch.zeros(self.num_envs, device=self.device),
                    torch.full((self.num_envs,), base_y, device=self.device),
                    gate_center_z
                ], dim=1)
                target = torch.stack([
                    torch.zeros(self.num_envs, device=self.device),
                    torch.zeros(self.num_envs, device=self.device),
                    gate_center_z
                ], dim=1)

                # Camera basis (right, up, forward)
                fwd = target - cam_pos
                fwd = fwd / (torch.norm(fwd, dim=1, keepdim=True) + 1e-8)
                up_world = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(1, 3).expand_as(fwd)
                right = torch.cross(fwd, up_world)
                right = right / (torch.norm(right, dim=1, keepdim=True) + 1e-8)
                up = torch.cross(right, fwd)

                # Transform drone position into camera coordinates
                pw = robot_position - cam_pos
                x_c = torch.sum(pw * right, dim=1)
                y_c = torch.sum(pw * up, dim=1)
                z_c = torch.sum(pw * fwd, dim=1)

                # Visibility checks within symmetric FOV (approx for D455)
                half_fov_rad = (87.0 * 3.141592653589793 / 180.0) * 0.5
                horiz_angle = torch.atan2(torch.abs(x_c), torch.clamp(z_c, min=1e-6))
                vert_angle = torch.atan2(torch.abs(y_c), torch.clamp(z_c, min=1e-6))
                visible = (z_c > 0.1) & (horiz_angle <= half_fov_rad) & (vert_angle <= half_fov_rad)

                # Graded score inside frustum: m = max(h/hfov, v/vfov); score = (1 - m)^alpha
                h_norm = torch.clamp(horiz_angle / half_fov_rad, 0.0, 1.0)
                v_norm = torch.clamp(vert_angle / half_fov_rad, 0.0, 1.0)
                m_norm = torch.maximum(h_norm, v_norm)
                fov_score = torch.pow(torch.clamp(1.0 - m_norm, min=0.0), fov_alpha)
                fov_reward = fov_mag * fov_score

                # Apply only for non-terminated envs and visible
                add_mask = visible & (~self.terminations)
                if torch.any(add_mask):
                    rewards[add_mask] = rewards[add_mask] + fov_reward[add_mask]
                    # Track episode totals
                    try:
                        self.episode_static_fov_visibility_reward[add_mask] += fov_reward[add_mask]
                    except Exception:
                        pass

                # Periodic debug infos
                if hasattr(self, 'num_task_steps') and (self.num_task_steps % 200 == 0):
                    try:
                        frac_visible = float(torch.mean(visible.float()).item())
                        avg_h = float(torch.mean(horiz_angle).item())
                        avg_v = float(torch.mean(vert_angle).item())
                        avg_score = float(torch.mean(fov_score).item())
                        self.infos["static_fov/visible_fraction"] = torch.tensor(frac_visible, dtype=torch.float32)
                        self.infos["static_fov/avg_horiz_angle_rad"] = torch.tensor(avg_h, dtype=torch.float32)
                        self.infos["static_fov/avg_vert_angle_rad"] = torch.tensor(avg_v, dtype=torch.float32)
                        self.infos["static_fov/avg_score"] = torch.tensor(avg_score, dtype=torch.float32)
                    except Exception:
                        pass

                    # Per-step env0 deep dive
                    try:
                        env0 = 0
                        if visible.shape[0] > env0:
                            vis0 = bool(visible[env0].item())
                            hdeg0 = float(horiz_angle[env0].item() * (180.0 / 3.141592653589793))
                            vdeg0 = float(vert_angle[env0].item() * (180.0 / 3.141592653589793))
                            xc0 = float(x_c[env0].item()); yc0 = float(y_c[env0].item()); zc0 = float(z_c[env0].item())
                            score0 = float(fov_score[env0].item())
                            # Current static camera yaw (if available)
                            try:
                                scm = getattr(self, 'static_camera_manager', None)
                                yaw_cur = float(scm.current_camera_angles[env0]) if (scm is not None and hasattr(scm, 'current_camera_angles') and len(scm.current_camera_angles) > env0) else 0.0
                            except Exception:
                                yaw_cur = 0.0
                            logger.warning(f" 🖼️ env0 FOV: visible={1 if vis0 else 0} h={hdeg0:.1f}° v={vdeg0:.1f}° | score={score0:.3f} | cam_yaw={yaw_cur:.1f}° | x_c={xc0:.2f}, y_c={yc0:.2f}, z_c={zc0:.2f}")
                    except Exception:
                        pass
        except Exception:
            pass
        # UPDATE EPISODE REWARD TRACKING: Track cumulative reward components
        self.update_episode_reward_tracking(obs_dict, rewards, crashes)
        # COMPREHENSIVE REWARD DEBUGGING: Print ALL reward components every 200 steps
        # Disabled by default via config flag `enable_comprehensive_reward_debug`
        if (
            hasattr(self, 'num_task_steps')
            and self.num_task_steps % 200 == 0
            and bool(getattr(self.task_config, 'enable_comprehensive_reward_debug', False))
        ):
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
            
            # Use adaptive gate center (z = bottom + center_height)
            gate_center_position = self.gate_position.clone()
            try:
                gate_center_position[:, 2] = gate_center_position[:, 2] + self.gate_center_height
            except Exception:
                pass
            gate_distance = torch.norm(robot_position - gate_center_position, dim=1)
            gate_approach_reward = exponential_reward_function(
                self.task_config.reward_parameters["gate_approach_reward_magnitude"],
                0.5,
                gate_distance,
            )
            
            # Gate alignment (finer piecewise by lateral offset from gate center)
            gate_alignment_reward = torch.zeros_like(gate_distance)
            dx = torch.abs(robot_position[:, 0] - self.gate_position[:, 0])
            # Thresholds proportional to current gate width (denser piecewise bins)
            t00 = self.gate_width * 0.01
            t0  = self.gate_width * 0.02
            t1  = self.gate_width * 0.04
            t2  = self.gate_width * 0.06
            t3  = self.gate_width * 0.08
            t4  = self.gate_width * 0.10
            t5  = self.gate_width * 0.12
            t6  = self.gate_width * 0.15
            t7  = self.gate_width * 0.20
            t8  = self.gate_width * 0.25
            t9  = self.gate_width * 0.30
            t10 = self.gate_width * 0.40
            t11 = self.gate_width * 0.50
            mag = self.task_config.reward_parameters["gate_alignment_reward_magnitude"]
            gate_alignment_reward[dx <= t00] = 1.00 * mag
            gate_alignment_reward[(dx > t00) & (dx <= t0)] = 0.97 * mag
            gate_alignment_reward[(dx > t0) & (dx <= t1)]  = 0.94 * mag
            gate_alignment_reward[(dx > t1) & (dx <= t2)]  = 0.90 * mag
            gate_alignment_reward[(dx > t2) & (dx <= t3)]  = 0.85 * mag
            gate_alignment_reward[(dx > t3) & (dx <= t4)]  = 0.80 * mag
            gate_alignment_reward[(dx > t4) & (dx <= t5)]  = 0.72 * mag
            gate_alignment_reward[(dx > t5) & (dx <= t6)]  = 0.65 * mag
            gate_alignment_reward[(dx > t6) & (dx <= t7)]  = 0.55 * mag
            gate_alignment_reward[(dx > t7) & (dx <= t8)]  = 0.45 * mag
            gate_alignment_reward[(dx > t8) & (dx <= t9)]  = 0.35 * mag
            gate_alignment_reward[(dx > t9) & (dx <= t10)] = 0.25 * mag
            gate_alignment_reward[(dx > t10) & (dx <= t11)] = 0.15 * mag
            
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
            # mult_factor = 1.0  # Disabled version (keep for quick ablation)
            avg_total_reward = torch.mean(rewards).item()
            # Use the effective multiplier factor computed earlier in this step
            try:
                mult_factor = float(getattr(self, '_curriculum_multiplier_factor', 1.0))
            except Exception:
                mult_factor = 1.0
            avg_pos_reward = torch.mean(mult_factor * pos_reward).item()
            avg_very_close = torch.mean(mult_factor * very_close_reward).item()
            avg_getting_closer = torch.mean(mult_factor * getting_closer_reward).item()
            avg_gate_approach = torch.mean(mult_factor * gate_approach_reward).item()
            avg_gate_alignment = torch.mean(mult_factor * gate_alignment_reward).item()
            avg_camera_facing = torch.mean(mult_factor * camera_facing_reward).item()
            avg_action_penalty = torch.mean(total_action_penalty).item()
            # Boundary violation penalty: calculate for debugging (same logic as torchscript)
            try:
                # Use consistent fixed y-margin for boundary check (same as TorchScript and episode tracker)
                y_margin = 0.2
                behind_gate_mask = (robot_position[:, 1] > (self.gate_position[:, 1] + y_margin))
                gate_passage_width_tolerance = self.gate_width * 0.6
                gate_min_height = self.gate_position[:, 2] + self.gate_height * 0.1
                gate_max_height = self.gate_position[:, 2] + self.gate_height * 0.9
                within_passage_window = (
                    (torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < gate_passage_width_tolerance)
                    & (robot_position[:, 2] > gate_min_height)
                    & (robot_position[:, 2] < gate_max_height)
                )
                misaligned_cross_mask = behind_gate_mask & (~within_passage_window) & (~self.gate_passed)
                boundary_violation_penalty = torch.zeros_like(gate_distance)
                boundary_violation_penalty[misaligned_cross_mask] = -50.0
                avg_boundary_penalty = torch.mean(boundary_violation_penalty).item()
            except Exception:
                avg_boundary_penalty = 0.0
            avg_distance = torch.mean(dist).item()
            avg_gate_distance = torch.mean(gate_distance).item()
            avg_camera_alignment = torch.mean(camera_gate_alignment).item()
            # Static FOV (recompute shaped-average for logging)
            try:
                fov_mag = float(self.task_config.reward_parameters.get("static_fov_visibility_reward_magnitude", 0.0))
            except Exception:
                fov_mag = 0.0
            avg_static_fov_reward = 0.0
            if fov_mag != 0.0:
                try:
                    parent = getattr(self, 'sim_env', None)
                    gtd = parent.global_tensor_dict if (parent is not None and hasattr(parent, 'global_tensor_dict')) else {}
                except Exception:
                    gtd = {}
                try:
                    base_y = float(os.environ.get('SF_STATIC_CAMERA_BASE_Y', gtd.get('static_camera/base_y', -3.0)))
                except Exception:
                    base_y = -3.0
                try:
                    base_z_env = os.environ.get('SF_STATIC_CAMERA_BASE_Z', None)
                    if base_z_env is None:
                        base_z_env = gtd.get('static_camera/base_z', 1.5)
                    adaptive_z = isinstance(base_z_env, str) and base_z_env.strip().lower() == 'adaptive'
                except Exception:
                    adaptive_z = False
                if adaptive_z:
                    gate_center_z = self.gate_center_height
                else:
                    gate_center_z = torch.full((self.num_envs,), 1.5, device=self.device)
                cam_pos = torch.stack([
                    torch.zeros(self.num_envs, device=self.device),
                    torch.full((self.num_envs,), base_y, device=self.device),
                    gate_center_z
                ], dim=1)
                target = torch.stack([
                    torch.zeros(self.num_envs, device=self.device),
                    torch.zeros(self.num_envs, device=self.device),
                    gate_center_z
                ], dim=1)
                fwd = target - cam_pos
                fwd = fwd / (torch.norm(fwd, dim=1, keepdim=True) + 1e-8)
                up_world = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(1, 3).expand_as(fwd)
                right = torch.cross(fwd, up_world); right = right / (torch.norm(right, dim=1, keepdim=True) + 1e-8)
                up = torch.cross(right, fwd)
                pw = robot_position - cam_pos
                x_c = torch.sum(pw * right, dim=1); y_c = torch.sum(pw * up, dim=1); z_c = torch.sum(pw * fwd, dim=1)
                half_fov_rad = (87.0 * 3.141592653589793 / 180.0) * 0.5
                horiz_angle = torch.atan2(torch.abs(x_c), torch.clamp(z_c, min=1e-6))
                vert_angle = torch.atan2(torch.abs(y_c), torch.clamp(z_c, min=1e-6))
                visible = (z_c > 0.1) & (horiz_angle <= half_fov_rad) & (vert_angle <= half_fov_rad)
                h_norm = torch.clamp(horiz_angle / half_fov_rad, 0.0, 1.0)
                v_norm = torch.clamp(vert_angle / half_fov_rad, 0.0, 1.0)
                m_norm = torch.maximum(h_norm, v_norm)
                try:
                    fov_alpha = float(self.task_config.reward_parameters.get("static_fov_visibility_exponent", 2.0))
                except Exception:
                    fov_alpha = 2.0
                fov_score = torch.pow(torch.clamp(1.0 - m_norm, min=0.0), fov_alpha)
                avg_static_fov_reward = float(torch.mean(fov_mag * fov_score).item())
            
            logger.warning("="*80)
            logger.warning(f"🔍 COMPREHENSIVE REWARD BREAKDOWN (Step {self.num_task_steps}):")
            logger.warning(f"  📊 TOTAL REWARD:           {avg_total_reward:.3f}")
            # Print VAE latent statistics alongside reward breakdown for clear visibility
            try:
                if hasattr(self, 'task_obs') and isinstance(self.task_obs, dict) and 'observations' in self.task_obs:
                    obs_all = self.task_obs['observations']
                    if obs_all.shape[1] >= 150:
                        d_lat = obs_all[:, 22:86]
                        s_lat = obs_all[:, 86:150]
                        d_mu = float(torch.mean(d_lat).item()); s_mu = float(torch.mean(s_lat).item())
                        d_std = float(torch.std(d_lat).item()); s_std = float(torch.std(s_lat).item())
                        d_norm = float(torch.linalg.norm(d_lat).item()); s_norm = float(torch.linalg.norm(s_lat).item())
                        ratio = s_norm / (d_norm + 1e-6)
                        logger.warning(
                            f"  🔬 VAE STATS: norm_ratio={ratio:.3f} | "
                            f"drone(mu={d_mu:.3f}, std={d_std:.3f}) | static(mu={s_mu:.3f}, std={s_std:.3f})"
                        )
            except Exception:
                pass
            logger.warning(f"  📍 Position Reward:        {avg_pos_reward:.3f} (dist: {avg_distance:.2f}m)")
            logger.warning(f"  🎯 Very Close Reward:      {avg_very_close:.3f}")
            logger.warning(f"  ⬆️  Getting Closer:         {avg_getting_closer:.3f}")
            logger.warning(f"  🚪 Gate Approach:          {avg_gate_approach:.3f} (gate_dist: {avg_gate_distance:.2f}m)")
            logger.warning(f"  ✅ Gate Alignment:         {avg_gate_alignment:.3f}")
            logger.warning(f"  📹 Camera Facing:          {avg_camera_facing:.3f} (align: {avg_camera_alignment:.3f})")
            if fov_mag != 0.0:
                logger.warning(f"  🖼️ Static FOV Reward:      {avg_static_fov_reward:.3f}")
            logger.warning(f"  🎮 Action Penalty:         {avg_action_penalty:.3f}")
            # Time/timeout penalties (averages)
            try:
                avg_time_pen = float(torch.mean(self.episode_time_penalty).item())
            except Exception:
                avg_time_pen = 0.0
            try:
                avg_timeout_pen = float(torch.mean(self.episode_timeout_penalty).item())
            except Exception:
                avg_timeout_pen = 0.0
            logger.warning(f"  ⏱️ Time Penalty (avg):     {avg_time_pen:.3f}")
            logger.warning(f"  ⌛ Timeout Penalty (avg):  {avg_timeout_pen:.3f}")
            logger.warning(f"  ⛔ Boundary Violation:     {avg_boundary_penalty:.3f}")
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
                # New: time-related penalties over last 10 episodes
                if 'time_penalty' in avg_episode_data:
                    logger.warning(f"  ⏱️ Time Penalties:         {avg_episode_data['time_penalty']:.1f}")
                if 'timeout_penalty' in avg_episode_data:
                    logger.warning(f"  ⌛ Timeout Penalties:      {avg_episode_data['timeout_penalty']:.1f}")
                if 'boundary_violation_penalty' in avg_episode_data:
                    logger.warning(f"  ⛔ Boundary Violations:    {avg_episode_data['boundary_violation_penalty']:.1f}")
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
            try:
                avg_current_boundary_penalty = torch.mean(self.episode_boundary_violation_penalty).item()
            except Exception:
                avg_current_boundary_penalty = 0.0
            avg_current_collision_penalty = torch.mean(self.episode_collision_penalty).item()
            avg_current_episode_length = torch.mean(self.episode_lengths).item()
            
            current_total = (avg_current_pos + avg_current_very_close + avg_current_getting_closer + 
                           avg_current_gate_approach + avg_current_gate_alignment + avg_current_camera_facing + 
                           avg_current_action_penalty + avg_current_collision_penalty + avg_current_boundary_penalty)
            
            logger.warning(f"  🔄 Current Episode Total:  {current_total:.1f} (avg across 16 envs)")
            logger.warning(f"  📍 Position So Far:        {avg_current_pos:.1f}")
            logger.warning(f"  ⬆️  Getting Closer So Far:  {avg_current_getting_closer:.1f}")
            logger.warning(f"  🚪 Gate Approach So Far:   {avg_current_gate_approach:.1f}")
            logger.warning(f"  ✅ Gate Alignment So Far:  {avg_current_gate_alignment:.1f}")
            logger.warning(f"  📹 Camera Facing So Far:   {avg_current_camera_facing:.1f}")
            logger.warning(f"  💥 Collision Penalties:    {avg_current_collision_penalty:.1f}")
            logger.warning(f"  📏 Steps So Far:           {avg_current_episode_length:.0f}")
            
            logger.warning("="*80)
        
        # Reward outlier logging to catch negative spikes
        try:
            thr = float(getattr(self.task_config, 'reward_outlier_threshold', -180.0))
        except Exception:
            thr = -180.0
        try:
            if torch.any(rewards < thr):
                _bad = torch.nonzero(rewards < thr, as_tuple=False).squeeze(-1)
                if _bad.numel() > 0:
                    _limit = int(getattr(self.task_config, 'reward_outlier_log_limit_per_step', 8))
                    _s = _bad[:_limit]
                    logger.warning(f"[RewardOutlier] envs={_s.tolist()} rewards={rewards[_s].tolist()} crashes={crashes[_s].tolist()}")
                    _dist = torch.norm(self.pos_error_vehicle_frame[_s], dim=1)
                    _y = obs_dict['robot_position'][_s, 1]
                    # Also log boundary violation mask to see if it caused spikes
                    try:
                        _bv = boundary_violation_one_shot_mask[_s].tolist()
                    except Exception:
                        _bv = []
                    logger.warning(f"[RewardOutlier] dist={_dist.tolist()} y={_y.tolist()} boundary_violation={_bv}")
        except Exception:
            pass
        
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
        # Early exit for testing/eval with forced level: lock level and skip progression
        try:
            forced = os.environ.get('SF_FORCE_CURRICULUM_LEVEL', None)
            if forced is None:
                forced = getattr(self.task_config, 'force_curriculum_level', None)
            if forced is not None and str(forced).lower() != 'no':
                eval_stretch_enabled = bool(getattr(self.task_config.curriculum, 'eval_stretch_enabled', False))
                effective_max = (
                    int(getattr(self.task_config.curriculum, 'eval_stretch_end_level', self.task_config.curriculum.max_level))
                    if eval_stretch_enabled else self.task_config.curriculum.max_level
                )
                self.curriculum_level = int(forced)
                self.curriculum_level = min(
                    max(self.curriculum_level, self.task_config.curriculum.min_level),
                    effective_max,
                )
                if hasattr(self, 'sim_env') and hasattr(self.sim_env, 'global_tensor_dict'):
                    self.sim_env.global_tensor_dict["curriculum_level"] = int(self.curriculum_level)
                try:
                    self.obs_dict["curriculum_level"] = self.curriculum_level
                except Exception:
                    pass
                try:
                    self.max_curriculum_level_reached = max(getattr(self, 'max_curriculum_level_reached', self.curriculum_level), self.curriculum_level)
                except Exception:
                    pass
                return
        except Exception:
            pass
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
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Thresholds: increase>{self.task_config.curriculum.success_rate_for_increase:.3f}, decrease<{self.task_config.curriculum.success_rate_for_decrease:.3f}")
            # Track cooldown state
            if not hasattr(self, '_curriculum_cooldown'): self._curriculum_cooldown = 0
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Cooldown windows remaining: {self._curriculum_cooldown}")
            # Maintain per-window success history (trim to last 3 windows)
            try:
                sr_float = float(success_rate.item()) if hasattr(success_rate, 'item') else float(success_rate)
            except Exception:
                sr_float = float(success_rate)
            if not hasattr(self, '_success_window_history'):
                self._success_window_history = []
            self._success_window_history.append(sr_float)
            if len(self._success_window_history) > 3:
                self._success_window_history.pop(0)
            # Compute current-window success and 3-window average (including current)
            s_t = sr_float
            if len(self._success_window_history) >= 3:
                avg3 = sum(self._success_window_history[-3:]) / 3.0
            else:
                # Use available windows until 3 are accumulated
                denom = max(1, len(self._success_window_history))
                avg3 = sum(self._success_window_history) / denom
            self.infos["curriculum/success_window_s_t"] = torch.tensor(s_t, dtype=torch.float32)
            self.infos["curriculum/success_avg3"] = torch.tensor(avg3, dtype=torch.float32)

            action_msg = "LEVEL UNCHANGED"
            # Respect cooldown
            if self._curriculum_cooldown > 0:
                self._curriculum_cooldown -= 1
                action_msg = f"LEVEL HOLD (cooldown {self._curriculum_cooldown} left)"
            else:
                # Check only at cooldown boundary
                inc_threshold = float(self.task_config.curriculum.success_rate_for_increase)
                avg3_threshold = float(getattr(self.task_config.curriculum, 'avg3_success_for_increase', 0.50))
                if (len(self._success_window_history) >= 3) and (s_t >= inc_threshold) and (avg3 >= avg3_threshold):
                    self.curriculum_level += self.task_config.curriculum.increase_step
                    self.max_curriculum_level_reached = max(self.max_curriculum_level_reached, self.curriculum_level)
                    self._curriculum_cooldown = getattr(self.task_config.curriculum, 'cooldown_windows', 0)
                    action_msg = (
                        f"LEVEL INCREASED: {old_level} -> {self.curriculum_level} "
                        f"(s_t {s_t:.3f} >= {inc_threshold:.2f} and avg3 {avg3:.3f} >= {avg3_threshold:.2f})"
                    )
                elif success_rate < self.task_config.curriculum.success_rate_for_decrease and self.curriculum_level > self.task_config.curriculum.min_level:
                    self.curriculum_level -= self.task_config.curriculum.decrease_step
                    self._curriculum_cooldown = getattr(self.task_config.curriculum, 'cooldown_windows', 0)
                    action_msg = f"LEVEL DECREASED: {old_level} -> {self.curriculum_level} (SR {success_rate:.3f} < threshold)"
            # Apply optional maximum cap without changing per-level scaling
            try:
                import os
                cap_env = os.environ.get('SF_MAX_CURRICULUM_LEVEL', None)
                cap_cfg = getattr(self.task_config, 'max_curriculum_level', None)
                cap = int(cap_env) if cap_env is not None else (int(cap_cfg) if cap_cfg is not None else None)
                if cap is not None:
                    if self.curriculum_level > cap:
                        self.curriculum_level = cap
                        action_msg = f"LEVEL CAPPED at {cap} (progression halted above cap)"
            except Exception:
                pass
            # Apply optional minimum start level (training only; no effect in inference)
            try:
                min_env = os.environ.get('SF_MIN_CURRICULUM_LEVEL', None)
                if min_env is not None:
                    min_cap = int(min_env)
                    if self.curriculum_level < min_cap:
                        self.curriculum_level = min_cap
                        action_msg = f"LEVEL RAISED to start min {min_cap}"
            except Exception:
                pass
            # Honor forced curriculum level: override and freeze progression
            try:
                forced = os.environ.get('SF_FORCE_CURRICULUM_LEVEL', None)
                if forced is None:
                    forced = getattr(self.task_config, 'force_curriculum_level', None)
                if forced is not None:
                    self.curriculum_level = int(forced)
                    action_msg = f"LEVEL FORCED: {self.curriculum_level} (progression disabled)"
                    # Reset aggregates to avoid immediate re-evaluation noise
                    self.success_aggregate = 0; self.crashes_aggregate = 0; self.timeouts_aggregate = 0
            except Exception:
                pass
            self.log_curriculum_update(f"[CURRICULUM UPDATE] {action_msg}")

            # Clamp curriculum_level to valid range (honor eval stretch end level if enabled)
            eval_stretch_enabled = bool(getattr(self.task_config.curriculum, 'eval_stretch_enabled', False))
            effective_max = (
                int(getattr(self.task_config.curriculum, 'eval_stretch_end_level', self.task_config.curriculum.max_level))
                if eval_stretch_enabled else self.task_config.curriculum.max_level
            )
            self.curriculum_level = min(
                max(self.curriculum_level, self.task_config.curriculum.min_level),
                effective_max,
            )
            self.obs_dict["curriculum_level"] = self.curriculum_level
            
            # Propagate curriculum level to env manager for gate unlocking
            if hasattr(self, 'sim_env') and hasattr(self.sim_env, 'global_tensor_dict'):
                # Only update the value; gate selection will occur on reset_idx
                self.sim_env.global_tensor_dict["curriculum_level"] = int(self.curriculum_level)
            
            # ===== MULTI-ASPECT CURRICULUM APPLICATION =====
            
            # 1. OBSTACLE COUNT PROGRESSION: Apply new obstacle count behind gate
            try:
                obs_dis = bool(self.sim_env.global_tensor_dict.get('obstacles_randomization/disabled', False))
            except Exception:
                obs_dis = False
            if obs_dis:
                try:
                    obstacles_behind_gate = int(self.sim_env.global_tensor_dict.get('obstacles_randomization/fixed_count', 0))
                except Exception:
                    obstacles_behind_gate = 0
            else:
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
            # Report yaw sweep status in curriculum update debug (takes precedence over orientation randomization)
            try:
                yaw_enabled = str(os.environ.get('SF_ENABLE_STATIC_CAMERA_YAW_SWEEP', 'false')).lower() == 'true'
                yaw_speed = float(os.environ.get('SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG', '10.0'))
            except Exception:
                yaw_enabled = False
                yaw_speed = 10.0
            # Determine dynamic camera effective state (needed below)
            try:
                dyn_cfg = getattr(self.task_config.curriculum, 'enable_dynamic_camera_following', False)
                dyn_dis = bool(self.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False))
                dynamic_effective = bool(dyn_cfg and not dyn_dis)
            except Exception:
                dynamic_effective = False
            # Report sweep with effective status and orientation/dynamic interactions
            if yaw_enabled and not dynamic_effective:
                self.log_curriculum_update(f"   3. STATIC CAMERA YAW SWEEP: ENABLED (speed={yaw_speed:.1f} deg/s)")
            elif yaw_enabled and dynamic_effective:
                self.log_curriculum_update(f"   3. STATIC CAMERA YAW SWEEP: ENABLED but IGNORED (dynamic camera active)")
            else:
                self.log_curriculum_update(f"   3. STATIC CAMERA YAW SWEEP: DISABLED")
            if obs_dis:
                self.log_curriculum_update(f"   1. OBSTACLES: fixed to {obstacles_behind_gate} behind gate (total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)")
            else:
                self.log_curriculum_update(f"   1. OBSTACLES: {obstacles_behind_gate} behind gate (total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)")
            try:
                baseline_level = int(getattr(self.task_config.curriculum, 'min_level', 3))
                pos_dis = bool(self.sim_env.global_tensor_dict.get('spawn_randomization/position_disabled', False))
                yaw_dis = bool(self.sim_env.global_tensor_dict.get('spawn_randomization/orientation_disabled', False))
                sr_active = self.task_config.curriculum.get_spawn_ranges(self.curriculum_level)
                sr_base = self.task_config.curriculum.get_spawn_ranges(baseline_level)
                sr_use = {
                    'x_half_span_m': sr_base['x_half_span_m'] if pos_dis else sr_active['x_half_span_m'],
                    'y_center_m':    sr_base['y_center_m']    if pos_dis else sr_active['y_center_m'],
                    'y_half_span_m': sr_base['y_half_span_m'] if pos_dis else sr_active['y_half_span_m'],
                    'z_center_m':    sr_base['z_center_m']    if pos_dis else sr_active['z_center_m'],
                    'z_half_span_m': sr_base['z_half_span_m'] if pos_dis else sr_active['z_half_span_m'],
                    'yaw_abs_rad':   sr_base['yaw_abs_rad']   if yaw_dis else sr_active['yaw_abs_rad'],
                }
                if pos_dis or yaw_dis:
                    status_pos = "DISABLED" if pos_dis else "ENABLED"
                    status_yaw = "DISABLED" if yaw_dis else "ENABLED"
                    self.log_curriculum_update(f"   2. SPAWN RANDOMIZATION: position={status_pos}, orientation={status_yaw}")
                self.log_curriculum_update(
                    f"   2. SPAWN: X∈[{(-sr_use['x_half_span_m']):.1f}, {(+sr_use['x_half_span_m']):.1f}] m, "
                    f"Y∈[{(sr_use['y_center_m']-sr_use['y_half_span_m']):.1f}, {(sr_use['y_center_m']+sr_use['y_half_span_m']):.1f}] m, "
                    f"Z∈[{(sr_use['z_center_m']-sr_use['z_half_span_m']):.1f}, {(sr_use['z_center_m']+sr_use['z_half_span_m']):.1f}] m; yaw ±{(sr_use['yaw_abs_rad']*57.2958):.1f}°"
                )
            except Exception as e:
                self.log_curriculum_update(f"   2. SPAWN: (fallback) Using fixed LMF2 config due to: {e}")
            # Get current randomized angle for first environment (representative)
            current_angle = 0.0
            if hasattr(self, 'static_camera_manager') and hasattr(self.static_camera_manager, 'current_camera_angles'):
                current_angle = self.static_camera_manager.current_camera_angles[0] if self.static_camera_manager.current_camera_angles else 0.0
            # Report static camera orientation randomization status (only relevant when yaw sweep is DISABLED)
            try:
                cam_orient_disabled = bool(self.sim_env.global_tensor_dict.get('static_camera_randomization/orientation_disabled', False))
            except Exception:
                cam_orient_disabled = False
            if yaw_enabled and not dynamic_effective:
                self.log_curriculum_update(f"   4. CAMERA ANGLE: overridden by yaw sweep (env0 current: {current_angle:.1f}°)")
            elif dynamic_effective:
                self.log_curriculum_update(f"   4. CAMERA ANGLE: suppressed (dynamic camera following active)")
            elif cam_orient_disabled:
                self.log_curriculum_update(f"   4. CAMERA ANGLE: randomization DISABLED, fixed at 0.0° (env0: {current_angle:.1f}°)")
            else:
                self.log_curriculum_update(f"   4. CAMERA ANGLE: ±{self.max_camera_angle:.1f}deg max range, env0: {current_angle:.1f}deg (fixed per episode)")
            
            # 4. GATE SIZE UNLOCKS (Curriculum-gated randomization) or Fixed (ablation)
            if hasattr(self.sim_env, 'global_tensor_dict'):
                gate_names = []
                if len(self.sim_env.global_tensor_dict.get("gate_variant_names_per_env", [])) > 0:
                    gate_names = self.sim_env.global_tensor_dict["gate_variant_names_per_env"][0]
                # Report fixed mode if enabled
                disable_flag = self.sim_env.global_tensor_dict.get('gate_randomization/disabled', False)
                try:
                    if hasattr(disable_flag, 'item'):
                        disable_flag = bool(disable_flag.item())
                    else:
                        disable_flag = bool(disable_flag)
                except Exception:
                    disable_flag = False
                if disable_flag:
                    try:
                        fixed_scale = self.sim_env.global_tensor_dict.get('gate_randomization/fixed_scale_percent', 100)
                        if hasattr(fixed_scale, 'item'):
                            fixed_scale = int(fixed_scale.item())
                        else:
                            fixed_scale = int(fixed_scale)
                    except Exception:
                        fixed_scale = 100
                    self.log_curriculum_update(f"   4. GATE SIZE: randomization disabled, fixed scale = {fixed_scale}%")
                else:
                    # Compute linear threshold from 80 -> 60 over levels 3..23
                    # If EVAL_STRETCH_ENABLED, extend further to 50% by eval_end_level
                    import os as _os
                    stretch_enabled = _os.environ.get("EVAL_STRETCH_ENABLED", "0").strip() in ("1", "true", "True")
                    eval_end = int(_os.environ.get("EVAL_STRETCH_END_LEVEL", str(getattr(self.task_config.curriculum, 'eval_stretch_end_level', 33))))
                    level = int(self.curriculum_level)
                    if level <= 3:
                        min_scale = 80
                    elif level <= 23:
                        frac = (level - 3) / (23 - 3)
                        raw = 80 - frac * (80 - 60)
                        min_scale = int((int(raw) // 2) * 2)
                    elif stretch_enabled:
                        # Extend 23->eval_end: 60% -> 50% linearly
                        if level >= eval_end:
                            min_scale = 50
                        else:
                            extra_frac = (level - 23) / max(1, (eval_end - 23))
                            raw = 60 - extra_frac * (60 - 50)
                            min_scale = int((int(raw) // 2) * 2)
                    else:
                        min_scale = 60
                    if min_scale < 50:
                        min_scale = 50
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
            try:
                cam_noise_disabled = bool(self.sim_env.global_tensor_dict.get('camera_randomization/noise_disabled', False))
            except Exception:
                cam_noise_disabled = False
            # Per-camera overrides for noise (presence-based overrides)
            try:
                gtd = getattr(self.sim_env, 'global_tensor_dict', {})
                drone_noise_key_present = 'camera_randomization/drone_noise_disabled' in gtd
                static_noise_key_present = 'camera_randomization/static_noise_disabled' in gtd
                drone_noise_flag = bool(gtd.get('camera_randomization/drone_noise_disabled', False)) if drone_noise_key_present else cam_noise_disabled
                static_noise_flag = bool(gtd.get('camera_randomization/static_noise_disabled', False)) if static_noise_key_present else cam_noise_disabled
            except Exception:
                drone_noise_flag = cam_noise_disabled
                static_noise_flag = cam_noise_disabled
            # Level-3 fallbacks when disabled
            d_std_min, d_drop_min = self.task_config.curriculum.get_camera_noise(3)
            eff_drone_std = camera_gaussian_std if not drone_noise_flag else d_std_min
            eff_static_std = camera_gaussian_std if not static_noise_flag else d_std_min
            eff_drone_drop = camera_dropout_rate if not drone_noise_flag else d_drop_min
            eff_static_drop = camera_dropout_rate if not static_noise_flag else d_drop_min
            self.log_curriculum_update(
                f"   5. CAMERA NOISE: drone(std={eff_drone_std:.4f}, pixel_drop={eff_drone_drop*100:.1f}%), static(std={eff_static_std:.4f}, pixel_drop={eff_static_drop*100:.1f}%)"
            )
            
            # 6. CAMERA FRAME DROPOUT (entire-frame)
            fd = self.task_config.curriculum.get_camera_frame_dropout(self.curriculum_level)
            try:
                cam_fd_disabled = bool(self.sim_env.global_tensor_dict.get('camera_randomization/frame_dropout_disabled', False))
            except Exception:
                cam_fd_disabled = False
            # Per-camera overrides for frame dropout (presence-based overrides)
            try:
                gtd = getattr(self.sim_env, 'global_tensor_dict', {})
                drone_fd_key_present = 'camera_randomization/drone_frame_dropout_disabled' in gtd
                static_fd_key_present = 'camera_randomization/static_frame_dropout_disabled' in gtd
                drone_fd_flag = bool(gtd.get('camera_randomization/drone_frame_dropout_disabled', False)) if drone_fd_key_present else cam_fd_disabled
                static_fd_flag = bool(gtd.get('camera_randomization/static_frame_dropout_disabled', False)) if static_fd_key_present else cam_fd_disabled
            except Exception:
                drone_fd_flag = cam_fd_disabled
                static_fd_flag = cam_fd_disabled
            # After change: when disabled, show level-3 minimum totals instead of 0
            fd_min = self.task_config.curriculum.get_camera_frame_dropout(3)
            eff_drone_tot = fd['drone_total'] if not drone_fd_flag else fd_min['drone_total']
            eff_static_tot = fd['static_total'] if not static_fd_flag else fd_min['static_total']
            eff_drone_freeze = fd['drone_freeze'] if not drone_fd_flag else fd_min['drone_freeze']
            eff_drone_blank = fd['drone_blank'] if not drone_fd_flag else fd_min['drone_blank']
            eff_static_freeze = fd['static_freeze'] if not static_fd_flag else fd_min['static_freeze']
            eff_static_blank = fd['static_blank'] if not static_fd_flag else fd_min['static_blank']
            self.log_curriculum_update(
                f"   6. CAMERA FRAME DROPOUT: drone_total={eff_drone_tot*100:.1f}% (freeze {eff_drone_freeze*100:.1f}%, blank {eff_drone_blank*100:.1f}%), static_total={eff_static_tot*100:.1f}% (freeze {eff_static_freeze*100:.1f}%, blank {eff_static_blank*100:.1f}%)"
            )
            
            # 7. STATE NOISE (pose)
            if getattr(self.task_config.curriculum, "enable_state_noise", False):
                try:
                    state_noise_disabled = bool(self.sim_env.global_tensor_dict.get('state_randomization/noise_disabled', False))
                except Exception:
                    state_noise_disabled = False
                if state_noise_disabled:
                    self.log_curriculum_update("   7. STATE NOISE: DISABLED (all std=0)")
                else:
                    sn = self.task_config.curriculum.get_state_noise(self.curriculum_level)
                    self.log_curriculum_update(
                        f"   7. STATE NOISE: drone_pos_std={sn['drone_pos_std_m']:.4f} m, drone_orient_std={sn['drone_orient_std_rad']*57.2958:.3f} deg, "
                        f"static_pos_std={sn['static_pos_std_m']:.4f} m, static_orient_std={sn['static_orient_std_rad']*57.2958:.3f} deg"
                    )
            else:
                self.log_curriculum_update("   7. STATE NOISE: disabled")
            
            # 8. DYNAMIC CAMERA FOLLOWING
            dynamic_enabled = getattr(self.task_config.curriculum, 'enable_dynamic_camera_following', False)
            try:
                dynamic_disabled = bool(self.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False))
                config_overridden = bool(self.sim_env.global_tensor_dict.get('dynamic_camera_following/config_overridden', False))
            except Exception:
                dynamic_disabled = False
                config_overridden = False
            
            if dynamic_enabled and not dynamic_disabled:
                if config_overridden:
                    self.log_curriculum_update("   8. DYNAMIC CAMERA: ENABLED by flag (--enable_dynamic_camera_following=true) - camera follows drone with adaptive gate targeting")
                else:
                    self.log_curriculum_update("   8. DYNAMIC CAMERA: ENABLED (camera follows drone with adaptive gate targeting)")
            elif dynamic_enabled and dynamic_disabled:
                self.log_curriculum_update("   8. DYNAMIC CAMERA: DISABLED by flag (--disable_dynamic_camera_following=true)")
            else:
                if config_overridden:
                    self.log_curriculum_update("   8. DYNAMIC CAMERA: DISABLED by flag (--enable_dynamic_camera_following=false)")
                else:
                    self.log_curriculum_update("   8. DYNAMIC CAMERA: DISABLED (static camera mode - curriculum-based positioning)")
            
            # Curriculum multiplier debug (update block)
            try:
                cm_disabled = str(os.environ.get('SF_DISABLE_CURRICULUM_MULTIPLIER', 'false')).lower() == 'true'
            except Exception:
                cm_disabled = False
            if not cm_disabled:
                try:
                    cm_disabled = bool(getattr(self.task_config, 'disable_curriculum_multiplier', False))
                except Exception:
                    cm_disabled = False
            frac_eff = 0.0 if cm_disabled else float(self.curriculum_progress_fraction)
            factor = 1.0 + 0.5 * frac_eff
            self.log_curriculum_update(f"   8. CURRICULUM MULTIPLIER: {'DISABLED' if cm_disabled else 'ENABLED'} (factor={factor:.3f})")
            
            # ===== CURRICULUM DEBUGGING: Final state after update =====
            self.log_curriculum_update(f"[CURRICULUM UPDATE] FINAL STATE:")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Level: {self.curriculum_level} (range: {self.task_config.curriculum.min_level}-{self.task_config.curriculum.max_level})")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Max level reached: {self.max_curriculum_level_reached} (DECREASE ENABLED)")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Progress: {self.curriculum_progress_fraction:.3f}")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Obstacles behind gate: {obstacles_behind_gate} (total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)")
            self.log_curriculum_update(f"[CURRICULUM UPDATE]   Asset manager: Updated both obs_dict and global_tensor_dict with count {total_obstacles_in_env}")
            # Report spawn ablation status with effective ranges
            try:
                baseline_level = int(getattr(self.task_config.curriculum, 'min_level', 3))
                pos_dis = bool(self.sim_env.global_tensor_dict.get('spawn_randomization/position_disabled', False))
                yaw_dis = bool(self.sim_env.global_tensor_dict.get('spawn_randomization/orientation_disabled', False))
                sr_active = self.task_config.curriculum.get_spawn_ranges(self.curriculum_level)
                sr_base = self.task_config.curriculum.get_spawn_ranges(baseline_level)
                sr_use = {
                    'x_half_span_m': sr_base['x_half_span_m'] if pos_dis else sr_active['x_half_span_m'],
                    'y_center_m':    sr_base['y_center_m']    if pos_dis else sr_active['y_center_m'],
                    'y_half_span_m': sr_base['y_half_span_m'] if pos_dis else sr_active['y_half_span_m'],
                    'z_center_m':    sr_base['z_center_m']    if pos_dis else sr_active['z_center_m'],
                    'z_half_span_m': sr_base['z_half_span_m'] if pos_dis else sr_active['z_half_span_m'],
                    'yaw_abs_rad':   sr_base['yaw_abs_rad']   if yaw_dis else sr_active['yaw_abs_rad'],
                }
                status_pos = "DISABLED" if pos_dis else "ENABLED"
                status_yaw = "DISABLED" if yaw_dis else "ENABLED"
                self.log_curriculum_update(f"[CURRICULUM UPDATE]   Spawn randomization: position={status_pos}, orientation={status_yaw}")
                self.log_curriculum_update(
                    f"[CURRICULUM UPDATE]   Spawn ranges: X∈[{(-sr_use['x_half_span_m']):.1f}, {(+sr_use['x_half_span_m']):.1f}] m, "
                    f"Y∈[{(sr_use['y_center_m']-sr_use['y_half_span_m']):.1f}, {(sr_use['y_center_m']+sr_use['y_half_span_m']):.1f}] m, "
                    f"Z∈[{(sr_use['z_center_m']-sr_use['z_half_span_m']):.1f}, {(sr_use['z_center_m']+sr_use['z_half_span_m']):.1f}] m; yaw ±{(sr_use['yaw_abs_rad']*57.2958):.1f}°"
                )
            except Exception:
                self.log_curriculum_update(f"[CURRICULUM UPDATE]   Spawn difficulty: LMF2 config (fallback)")
            # When yaw sweep is enabled and dynamic camera is not active, suppress static camera angle randomization message
            try:
                yaw_enabled = str(os.environ.get('SF_ENABLE_STATIC_CAMERA_YAW_SWEEP', 'false')).lower() == 'true'
            except Exception:
                yaw_enabled = False
            dynamic_effective = False
            try:
                dyn_cfg = getattr(self.task_config.curriculum, 'enable_dynamic_camera_following', False)
                dyn_dis = bool(self.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False))
                dynamic_effective = bool(dyn_cfg and not dyn_dis)
            except Exception:
                dynamic_effective = False
            if yaw_enabled and not dynamic_effective:
                # Already logged as overridden by yaw sweep earlier
                pass
            else:
                self.log_curriculum_update(f"[CURRICULUM UPDATE]   Camera angle: ±{self.max_camera_angle:.1f}deg max range (randomized per episode reset, fixed during episode)")
            
            # ===== END CURRICULUM DEBUGGING =====
            
            # Add comprehensive curriculum metrics to infos for wandb logging
            self.infos["curriculum/level"] = torch.as_tensor(self.curriculum_level, dtype=torch.float32)
            self.infos["curriculum/progress"] = torch.as_tensor(self.curriculum_progress_fraction, dtype=torch.float32)
            self.infos["curriculum/success_rate"] = torch.as_tensor(success_rate, dtype=torch.float32)
            self.infos["curriculum/crash_rate"] = torch.as_tensor(crash_rate, dtype=torch.float32)
            self.infos["curriculum/timeout_rate"] = torch.as_tensor(timeout_rate, dtype=torch.float32)
            
            # Add curriculum metrics
            self.infos["curriculum/obstacles_behind_gate"] = torch.as_tensor(obstacles_behind_gate, dtype=torch.float32)
            self.infos["curriculum/total_assets"] = torch.as_tensor(total_obstacles_in_env, dtype=torch.float32)
            self.infos["curriculum/max_level_reached"] = torch.as_tensor(self.max_curriculum_level_reached, dtype=torch.float32)
            
            # Add camera noise metrics (D455 simulation) — report effective per-camera values in logs above
            self.infos["curriculum/camera_gaussian_std"] = torch.as_tensor(camera_gaussian_std, dtype=torch.float32)
            self.infos["curriculum/camera_dropout_rate"] = torch.as_tensor(camera_dropout_rate, dtype=torch.float32)
            # Per-camera effective values (respecting per-camera disable overrides) — level-3 fallback when disabled
            try:
                gtd = getattr(self.sim_env, 'global_tensor_dict', {})
                drone_noise_dis = bool(gtd.get('camera_randomization/drone_noise_disabled', False))
                static_noise_dis = bool(gtd.get('camera_randomization/static_noise_disabled', False))
                # Level-3 minimums
                d_std_min, d_drop_min = self.task_config.curriculum.get_camera_noise(3)
                # Effective Gaussian std per camera
                eff_gauss_drone = camera_gaussian_std if not drone_noise_dis else d_std_min
                eff_gauss_static = camera_gaussian_std if not static_noise_dis else d_std_min
                # Effective pixel dropout per camera
                eff_drop_drone = camera_dropout_rate if not drone_noise_dis else d_drop_min
                eff_drop_static = camera_dropout_rate if not static_noise_dis else d_drop_min
                self.infos["curriculum/camera_noise_drone_gaussian_std"] = torch.tensor(eff_gauss_drone, dtype=torch.float32)
                self.infos["curriculum/camera_noise_static_gaussian_std"] = torch.tensor(eff_gauss_static, dtype=torch.float32)
                self.infos["curriculum/camera_noise_drone_dropout_rate"] = torch.tensor(eff_drop_drone, dtype=torch.float32)
                self.infos["curriculum/camera_noise_static_dropout_rate"] = torch.tensor(eff_drop_static, dtype=torch.float32)
            except Exception:
                pass
            # Add camera frame dropout metrics (effective per-camera, with level-3 fallback when disabled)
            fd_sched = self.task_config.curriculum.get_camera_frame_dropout(self.curriculum_level)
            try:
                gtd = getattr(self.sim_env, 'global_tensor_dict', {})
                drone_fd_flag = bool(gtd.get('camera_randomization/drone_frame_dropout_disabled', False))
                static_fd_flag = bool(gtd.get('camera_randomization/static_frame_dropout_disabled', False))
            except Exception:
                drone_fd_flag = False
                static_fd_flag = False
            fd_min = self.task_config.curriculum.get_camera_frame_dropout(3)
            eff = {
                'drone_total':  fd_sched['drone_total']  if not drone_fd_flag else fd_min['drone_total'],
                'static_total': fd_sched['static_total'] if not static_fd_flag else fd_min['static_total'],
                'drone_freeze': fd_sched['drone_freeze'] if not drone_fd_flag else fd_min['drone_freeze'],
                'drone_blank':  fd_sched['drone_blank']  if not drone_fd_flag else fd_min['drone_blank'],
                'static_freeze':fd_sched['static_freeze']if not static_fd_flag else fd_min['static_freeze'],
                'static_blank': fd_sched['static_blank'] if not static_fd_flag else fd_min['static_blank'],
            }
            self.infos["curriculum/camera_frame_dropout_drone_total"] = torch.tensor(eff["drone_total"], dtype=torch.float32)
            self.infos["curriculum/camera_frame_dropout_static_total"] = torch.tensor(eff["static_total"], dtype=torch.float32)
            self.infos["curriculum/camera_frame_freeze_drone"] = torch.tensor(eff["drone_freeze"], dtype=torch.float32)
            self.infos["curriculum/camera_frame_blank_drone"] = torch.tensor(eff["drone_blank"], dtype=torch.float32)
            self.infos["curriculum/camera_frame_freeze_static"] = torch.tensor(eff["static_freeze"], dtype=torch.float32)
            self.infos["curriculum/camera_frame_blank_static"] = torch.tensor(eff["static_blank"], dtype=torch.float32)
            
            # Add camera angle metrics
            self.infos["curriculum/camera_max_angle"] = torch.tensor(self.max_camera_angle, dtype=torch.float32)
            # Use first environment's angle as representative for wandb tracking
            current_angle = 0.0
            if hasattr(self, 'static_camera_manager') and hasattr(self.static_camera_manager, 'current_camera_angles'):
                current_angle = self.static_camera_manager.current_camera_angles[0] if self.static_camera_manager.current_camera_angles else 0.0
            self.infos["curriculum/camera_current_angle"] = torch.tensor(current_angle, dtype=torch.float32)
            # Track ablation flag in infos
            try:
                cam_orient_disabled = bool(self.sim_env.global_tensor_dict.get('static_camera_randomization/orientation_disabled', False))
            except Exception:
                cam_orient_disabled = False
            self.infos["curriculum/camera_orientation_randomization_disabled"] = torch.tensor(1.0 if cam_orient_disabled else 0.0, dtype=torch.float32)
            
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
        # mult_factor = 1.0  # Disabled version (keep for quick ablation)
        
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
        # Use adaptive gate center (z = bottom + center_height)
        gate_center_position = self.gate_position.clone()
        try:
            gate_center_position[:, 2] = gate_center_position[:, 2] + self.gate_center_height
        except Exception:
            pass
        gate_distance = torch.norm(robot_position - gate_center_position, dim=1)
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
        # Gate the camera-facing reward: only before first crossing and while approaching (y below gate plane)
        approach_mask = (robot_position[:, 1] < self.gate_position[:, 1] - 0.1) & (~self.gate_passed)
        camera_facing_reward = camera_facing_reward * approach_mask.float()
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
        
        # Boundary violation penalty (episode tracking mirror of TorchScript path)
        try:
            # Use ENTIRE gate opening (±0.5 width, full height) and zero plane margin
            y_margin = 0.2
            behind_gate_mask = (robot_position[:, 1] > (self.gate_position[:, 1] + y_margin))
            full_width_tol = self.gate_width * 0.5
            full_min_h = self.gate_position[:, 2] + 0.0 * self.gate_height
            full_max_h = self.gate_position[:, 2] + 1.0 * self.gate_height
            within_passage_window = (
                (torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < full_width_tol)
                & (robot_position[:, 2] > full_min_h)
                & (robot_position[:, 2] < full_max_h)
            )
            misaligned_cross_mask = behind_gate_mask & (~within_passage_window) & (~self.gate_passed) & (~gate_passed_this_step)
            boundary_violation_penalty = torch.zeros_like(gate_distance)
            # One-shot penalty per episode: apply only on rising edge
            if not hasattr(self, '_bv_flag_episode'):
                self._bv_flag_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            rising_mask = misaligned_cross_mask & (~self._bv_flag_episode)
            boundary_violation_penalty[rising_mask] = -50.0
            # Update flag and force termination for violating envs
            self._bv_flag_episode = self._bv_flag_episode | misaligned_cross_mask
            if torch.any(rising_mask):
                # Use terminations for boundary violations (true MDP failure)
                self.terminations[rising_mask] = 1
                # env0 debug already prints above on rising edge
            self.episode_boundary_violation_penalty += boundary_violation_penalty
            # Print on rising edge for all envs that violated this step
            try:
                rising_envs = torch.nonzero(rising_mask, as_tuple=False).squeeze(-1)
                if rising_envs.numel() > 0:
                    for eid in rising_envs.tolist():
                        rx = float(robot_position[eid, 0].item())
                        ry = float(robot_position[eid, 1].item())
                        rz = float(robot_position[eid, 2].item())
                        gate_x = float(self.gate_position[eid, 0].item())
                        gate_y = float(self.gate_position[eid, 1].item())
                        x_off = abs(rx - gate_x)
                        gw = float(self.gate_width[eid].item() if hasattr(self.gate_width, 'shape') else self.gate_width)
                        gh = float(self.gate_height[eid].item() if hasattr(self.gate_height, 'shape') else self.gate_height)
                        tol = float(full_width_tol[eid].item() if hasattr(full_width_tol, 'shape') else full_width_tol)
                        zmin = float(full_min_h[eid].item() if hasattr(full_min_h, 'shape') else full_min_h)
                        zmax = float(full_max_h[eid].item() if hasattr(full_max_h, 'shape') else full_max_h)
                        logger.warning(
                            f"[Boundary] Env{eid} VIOLATION at step {self.num_task_steps}: pos=({rx:.3f},{ry:.3f},{rz:.3f}), "
                            f"gate_y={gate_y:.3f}, x_off={x_off:.3f} (tol={tol:.3f}), z_window=({zmin:.3f},{zmax:.3f}), "
                            f"gate_size=(w={gw:.3f}, h={gh:.3f})"
                        )
            except Exception:
                pass
        except Exception:
            pass
        
        # Track image rewards (from post_image_reward_addition)
        if hasattr(self, 'min_pixel_dist'):
            mag = self.task_config.reward_parameters.get("image_penalty_magnitude", 4.0)
            expo = self.task_config.reward_parameters.get("image_penalty_exponent", 1.0)
            image_rewards = -exponential_reward_function(
                float(mag), float(expo), self.min_pixel_dist[~self.terminations]
            )
            # Only add for non-terminated environments
            non_terminated_mask = ~self.terminations
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
                        self.episode_time_penalty[env_id] + 
                        self.episode_timeout_penalty[env_id] + 
                        self.episode_boundary_violation_penalty[env_id] + 
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
                    'boundary_violation_penalty': self.episode_boundary_violation_penalty[env_id].item(),
                    'time_penalty': self.episode_time_penalty[env_id].item(),
                    'timeout_penalty': self.episode_timeout_penalty[env_id].item(),
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
        self.episode_boundary_violation_penalty[env_ids] = 0
        self.episode_time_penalty[env_ids] = 0
        self.episode_timeout_penalty[env_ids] = 0
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
        # Ensure device exists for tensor ops (fix jitter sampling fallback)
        try:
            self.device = getattr(env_manager, 'device', None)
            if self.device is None:
                self.device = getattr(task_config, 'device', 'cpu')
        except Exception:
            self.device = 'cpu'
        
        # Gate position (will be updated dynamically based on actual gate positions)
        self.gate_position = [0.0, 0.0, 0.0]  # Default, overridden by adaptive positioning
        self.env_bounds = [[-4.0, -4.0, 0.0], [4.0, 4.0, 4.0]]  # Updated for gate_env bounds
        
        # Per-environment camera angle tracking - FIXED during each episode
        self.num_envs = len(self.env_handles)
        self.current_camera_angles = [0.0] * self.num_envs  # Track angle per environment
        # Debug tracking of last camera placement and target per env
        self.last_camera_pos = [(0.0, -3.0, 1.5) for _ in range(self.num_envs)]
        self.last_camera_target = [(0.0, 0.0, 1.5) for _ in range(self.num_envs)]
        self.last_angle_deg = [0.0 for _ in range(self.num_envs)]
        # Per-env placement randomization (translation + small Euler jitter)
        # Defaults are no jitter; can be enabled/configured via task_config
        self.static_cam_randomize = bool(getattr(task_config, 'static_camera_randomize_placement', False))
        self.static_cam_min_t = list(getattr(task_config, 'static_camera_min_translation', [0.0, 0.0, 0.0]))
        self.static_cam_max_t = list(getattr(task_config, 'static_camera_max_translation', [0.0, 0.0, 0.0]))
        self.static_cam_min_euler = list(getattr(task_config, 'static_camera_min_euler_deg', [0.0, 0.0, 0.0]))
        self.static_cam_max_euler = list(getattr(task_config, 'static_camera_max_euler_deg', [0.0, 0.0, 0.0]))
        # Storage for per-env jitter, refreshed on setup and per-episode resets
        self._trans_jitter = [(0.0, 0.0, 0.0) for _ in range(self.num_envs)]
        self._euler_jitter_deg = [(0.0, 0.0, 0.0) for _ in range(self.num_envs)]
        
        self._setup_static_camera()
    
    def get_average_camera_angle(self):
        """Get average camera angle across all environments for logging."""
        if not hasattr(self, 'current_camera_angles') or not self.current_camera_angles:
            return 0.0
        return sum(self.current_camera_angles) / len(self.current_camera_angles)
    
    def _setup_static_camera(self):
        """Setup static camera using Isaac Gym native camera API with D455 specifications."""
        logger.info("Setting up static camera for gate navigation...")
        
        # Headless mode is supported for camera sensors; proceed with GPU camera setup
        if self.task_config.headless:
            logger.info("Headless mode detected - proceeding with static camera GPU setup")
        
        try:
            # Camera properties (D455 depth camera specifications - match working example)
            camera_props = gymapi.CameraProperties()
            camera_props.width = 240  # Reduced depth resolution width
            camera_props.height = 135  # Reduced depth resolution height
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
                    # Do not fall back to synthetic or return; proceed and mark setup incomplete
            
            # Respect CLI/task_config base_y/base_z (with 'adaptive' support) for initial placement
            try:
                base_y = float(getattr(self.task_config, 'static_camera_base_y', float(os.environ.get('SF_STATIC_CAMERA_BASE_Y', -3.0))))
            except Exception:
                base_y = -3.0
            # Determine base Z spawning mode: numeric or 'adaptive' to gate center per env
            adaptive_z = False
            base_z_value = 1.5
            try:
                cfg_base_z = getattr(self.task_config, 'static_camera_base_z', None)
                if cfg_base_z is not None:
                    if isinstance(cfg_base_z, str) and cfg_base_z.strip().lower() == 'adaptive':
                        adaptive_z = True
                    else:
                        base_z_value = float(cfg_base_z)
                else:
                    env_base_z = os.environ.get('SF_STATIC_CAMERA_BASE_Z', None)
                    if env_base_z is not None and str(env_base_z).strip().lower() == 'adaptive':
                        adaptive_z = True
                    elif env_base_z is not None:
                        base_z_value = float(env_base_z)
            except Exception:
                pass

            # Try to read per-env adaptive gate center Z from global tensors (if available)
            try:
                gtd = self.env_manager.IGE_env.global_tensor_dict
                gate_center_per_env = gtd.get('gate/center_height_per_env', None)
            except Exception:
                gate_center_per_env = None

            # Set camera transform for each environment using configured positioning
            for i, (env_handle, cam_handle) in enumerate(zip(self.env_handles, self.camera_handles)):
                # Initialize per-env jitter for the first episode
                if self.static_cam_randomize:
                    try:
                        # Sample translation jitter uniformly between per-axis bounds
                        u = torch.rand(3, device=self.device)
                        tmin = torch.tensor(self.static_cam_min_t, device=self.device, dtype=torch.float32)
                        tmax = torch.tensor(self.static_cam_max_t, device=self.device, dtype=torch.float32)
                        t = (tmin + u * (tmax - tmin)).tolist()
                        # Sample Euler jitter (roll, pitch, yaw) in degrees
                        v = torch.rand(3, device=self.device)
                        emin = torch.tensor(self.static_cam_min_euler, device=self.device, dtype=torch.float32)
                        emax = torch.tensor(self.static_cam_max_euler, device=self.device, dtype=torch.float32)
                        e = (emin + v * (emax - emin)).tolist()
                        self._trans_jitter[i] = (float(t[0]), float(t[1]), float(t[2]))
                        self._euler_jitter_deg[i] = (float(e[0]), float(e[1]), float(e[2]))
                    except Exception:
                        self._trans_jitter[i] = (0.0, 0.0, 0.0)
                        self._euler_jitter_deg[i] = (0.0, 0.0, 0.0)
                if adaptive_z and gate_center_per_env is not None and i < len(gate_center_per_env):
                    env_base_z = float(gate_center_per_env[i].item())
                else:
                    env_base_z = float(base_z_value)
                # Apply translation jitter
                jx, jy, jz = self._trans_jitter[i]
                camera_pos = gymapi.Vec3(0.0 + jx, float(base_y) + jy, env_base_z + jz)
                # Look directly at the gate center height for that env
                camera_target = gymapi.Vec3(0.0, 0.0, env_base_z)
                self.gym.set_camera_location(cam_handle, env_handle, camera_pos, camera_target)
                logger.info(
                    f"Set static camera {i} to look from ({camera_pos.x}, {camera_pos.y}, {camera_pos.z}) toward ({camera_target.x}, {camera_target.y}, {camera_target.z})"
                )
                # Store for debug
                try:
                    self.last_camera_pos[i] = (float(camera_pos.x), float(camera_pos.y), float(camera_pos.z))
                    self.last_camera_target[i] = (float(camera_target.x), float(camera_target.y), float(camera_target.z))
                    self.last_angle_deg[i] = 0.0
                except Exception:
                    pass

            # Cache base values into global dict for downstream updates
            try:
                if hasattr(self.env_manager.IGE_env, 'global_tensor_dict'):
                    self.env_manager.IGE_env.global_tensor_dict['static_camera/base_y'] = float(base_y)
                    self.env_manager.IGE_env.global_tensor_dict['static_camera/base_z'] = ('adaptive' if adaptive_z else float(base_z_value))
            except Exception:
                pass

            if len(self.camera_handles) == len(self.env_handles):
                logger.info("✓ Static camera setup complete with configured positioning (base_y/base_z)")
                self.camera_setup_success = True
                self.use_synthetic_camera = False
            else:
                logger.error(f"Static camera setup incomplete: created {len(self.camera_handles)}/{len(self.env_handles)} cameras")
                self.camera_setup_success = False
                self.use_synthetic_camera = False
            
        except Exception as e:
            logger.warning(f"Static camera setup failed: {e}")
            self.camera_setup_success = False
            self.use_synthetic_camera = False
    
    def update_camera_positions(self, curriculum_level, env_ids):
        """Update static camera orientation ONLY for resetting environments."""
        if hasattr(self, 'use_synthetic_camera') and self.use_synthetic_camera:
            # In synthetic mode, update orientation per resetting env with spawn-aware logic
            from aerial_gym.config.task_config.navigation_task_config_gate import task_config
            max_angle_range, _, _ = task_config.curriculum.get_static_camera_difficulty(curriculum_level)
            # Read flags and robot positions
            try:
                parent = getattr(self, 'env_manager', None)
                disable_flag = False
                rp = None
                if parent is not None and hasattr(parent, 'global_tensor_dict'):
                    gtd = parent.global_tensor_dict
                    disable_flag = bool(gtd.get('static_camera_randomization/orientation_disabled', False))
                    rp = gtd.get('robot_position', None)
            except Exception:
                disable_flag = False
                rp = None
            import random, math
            horizontal_fov = 87.0
            half_fov = horizontal_fov * 0.5
            margin = 5.0
            for env_idx in env_ids:
                if env_idx < len(self.current_camera_angles):
                    if disable_flag or max_angle_range <= 0:
                        ang = 0.0
                    else:
                        # Camera at (0,-3) looks toward +Y (0°). Keep both gate (0°) and drone in FOV.
                        if rp is not None and env_idx < rp.shape[0]:
                            cam_x, cam_y = 0.0, -3.0
                            dx = float(rp[env_idx, 0].item()) - cam_x
                            dy = float(rp[env_idx, 1].item()) - cam_y
                            theta_r = math.degrees(math.atan2(dx, dy))
                            gate_low, gate_high = -half_fov + margin, half_fov - margin
                            rob_low, rob_high = theta_r - (half_fov - margin), theta_r + (half_fov - margin)
                            low = max(gate_low, rob_low, -max_angle_range)
                            high = min(gate_high, rob_high, max_angle_range)
                            if high > low:
                                ang = random.uniform(low, high)
                            else:
                                target = max(min(theta_r, gate_high), gate_low)
                                ang = max(-max_angle_range, min(max_angle_range, target))
                        else:
                            ang = random.uniform(-max_angle_range, max_angle_range)
                    self.current_camera_angles[env_idx] = ang
            logger.debug(f"Synthetic camera mode - updated angles for envs {env_ids.tolist()}")
            return
            
        if not self.camera_setup_success or len(self.camera_handles) == 0:
            return
        
        # Get maximum angle range. For yaw sweep feature we use fixed ±30° (curriculum-independent)
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        _max_angle_range, _, _ = task_config.curriculum.get_static_camera_difficulty(curriculum_level)
        FIXED_SWEEP_MAX_DEG = 15.0
        # Honor ablation flag from parent task/global dict and read robot positions
        try:
            parent = getattr(self, 'env_manager', None)
            disable_flag = False
            rp = None
            if parent is not None and hasattr(parent, 'global_tensor_dict'):
                gtd = parent.global_tensor_dict
                disable_flag = bool(gtd.get('static_camera_randomization/orientation_disabled', False))
                rp = gtd.get('robot_position', None)
        except Exception:
            disable_flag = False
            rp = None
        
        try:
            # Fixed camera base position with optional CLI overrides
            try:
                parent = getattr(self, 'env_manager', None)
                gtd = parent.global_tensor_dict if (parent is not None and hasattr(parent, 'global_tensor_dict')) else {}
            except Exception:
                gtd = {}
            try:
                base_y = float(os.environ.get('SF_STATIC_CAMERA_BASE_Y', gtd.get('static_camera/base_y', -3.0)))
            except Exception:
                base_y = -3.0
            # Determine base Z spawning mode: numeric or 'adaptive' to gate center
            try:
                base_z_env = os.environ.get('SF_STATIC_CAMERA_BASE_Z', None)
                if base_z_env is None:
                    base_z_env = gtd.get('static_camera/base_z', 1.5)
                # Accept string 'adaptive' (case-insensitive) to enable adaptive Z per env
                if isinstance(base_z_env, str) and base_z_env.strip().lower() == 'adaptive':
                    base_z = None  # Means adaptive per env
                else:
                    base_z = float(base_z_env)
            except Exception:
                base_z = 1.5
            # If base_z is None (adaptive per env), use a numeric placeholder for Vec3; per-env Z resolved below
            try:
                base_z_for_vec = 1.5 if base_z is None else float(base_z)
            except Exception:
                base_z_for_vec = 1.5
            base_camera_pos = gymapi.Vec3(0.0, base_y, base_z_for_vec)
            
            import math
            import random
            
            # Ensure per-env randomized sweep parameters exist
            try:
                if not hasattr(self, 'sweep_phase_offsets') or (len(getattr(self, 'sweep_phase_offsets', [])) != len(self.env_handles)):
                    self.sweep_phase_offsets = [0.0 for _ in range(len(self.env_handles))]
                    self.sweep_directions = [1.0 for _ in range(len(self.env_handles))]
            except Exception:
                pass
            
            # Update camera orientation ONLY for the specified environments (those resetting)
            for env_idx in env_ids:
                if env_idx >= len(self.env_handles) or env_idx >= len(self.camera_handles):
                    continue
                # Re-sample per-env jitter on reset for fresh episodes
                if self.static_cam_randomize:
                    try:
                        u = torch.rand(3, device=self.device)
                        tmin = torch.tensor(self.static_cam_min_t, device=self.device, dtype=torch.float32)
                        tmax = torch.tensor(self.static_cam_max_t, device=self.device, dtype=torch.float32)
                        t = (tmin + u * (tmax - tmin)).tolist()
                        v = torch.rand(3, device=self.device)
                        emin = torch.tensor(self.static_cam_min_euler, device=self.device, dtype=torch.float32)
                        emax = torch.tensor(self.static_cam_max_euler, device=self.device, dtype=torch.float32)
                        e = (emin + v * (emax - emin)).tolist()
                        self._trans_jitter[env_idx] = (float(t[0]), float(t[1]), float(t[2]))
                        self._euler_jitter_deg[env_idx] = (float(e[0]), float(e[1]), float(e[2]))
                    except Exception:
                        self._trans_jitter[env_idx] = (0.0, 0.0, 0.0)
                        self._euler_jitter_deg[env_idx] = (0.0, 0.0, 0.0)
                    
                # Optional: constant yaw sweep (±30°), curriculum-independent
                try:
                    parent = getattr(self, 'env_manager', None)
                    gtd = parent.global_tensor_dict if (parent is not None and hasattr(parent, 'global_tensor_dict')) else {}
                    sweep_enabled = str(gtd.get('static_camera/yaw_sweep_enabled', 'false')).lower() == 'true'
                    sweep_speed_deg = float(gtd.get('static_camera/yaw_sweep_speed_deg', 10.0))
                except Exception:
                    sweep_enabled = False
                    sweep_speed_deg = 10.0

                if sweep_enabled:
                    # Compute time-based angle: A(level)*sin(omega*t + phase).
                    # Linear amplitude schedule: 2° at level 3 → 19° at level 23; clamp outside.
                    start_level = 3
                    end_level = 23
                    A_min = 2.0
                    A_max = 19.0
                    if curriculum_level <= start_level:
                        A = A_min
                    elif curriculum_level >= end_level:
                        A = A_max
                    else:
                        frac = float(curriculum_level - start_level) / float(end_level - start_level)
                        A = A_min + frac * (A_max - A_min)
                    dt = 1.0/60.0
                    # Keep peak angular speed similar to baseline A0=50° when changing amplitude.
                    # For theta(t)=A*sin(ωt), peak speed = A*ω. Compensate ω by (A0/A).
                    A0 = 50.0
                    comp = (A0 / max(A, 1e-6))
                    # Additionally, increase sweep speed with curriculum level (1.0x -> 2.0x)
                    try:
                        from aerial_gym.config.task_config.navigation_task_config_gate import task_config as _tc2
                        # Respect eval stretch when enabled (eval only), otherwise cap at training max
                        try:
                            parent = getattr(self, 'env_manager', None)
                            gtd_local = parent.global_tensor_dict if (parent is not None and hasattr(parent, 'global_tensor_dict')) else {}
                        except Exception:
                            gtd_local = {}
                        eval_en = bool(gtd_local.get('eval_stretch_enabled', False))
                        min_lvl = int(getattr(_tc2.curriculum, 'min_level', 1))
                        max_lvl_cfg = int(getattr(_tc2.curriculum, 'max_level', min_lvl))
                        max_lvl_eval = int(getattr(_tc2.curriculum, 'eval_stretch_end_level', max_lvl_cfg))
                        max_lvl = max_lvl_eval if eval_en else max_lvl_cfg
                        level_clamped = max(min(curriculum_level, max_lvl), min_lvl)
                        denom = max(1, max_lvl - min_lvl)
                        level_frac = float(level_clamped - min_lvl) / float(denom)
                        speed_scale = 1.0 + level_frac
                    except Exception:
                        speed_scale = 1.0
                    sweep_speed_eff = sweep_speed_deg * speed_scale * comp
                    omega = (sweep_speed_eff * 3.14159 / 180.0) * dt
                    # Use global sim step as t and per-env small phase to desynchronize
                    sim_steps = 0
                    try:
                        steps_obj = gtd.get('sim_steps', 0)
                        # Support torch tensors or plain ints
                        if hasattr(steps_obj, 'shape') or hasattr(steps_obj, 'ndim'):
                            # Torch tensor
                            try:
                                if getattr(steps_obj, 'ndim', 0) == 0:
                                    sim_steps = int(steps_obj.item())
                                else:
                                    idx = env_idx if env_idx < steps_obj.shape[0] else 0
                                    sim_steps = int(steps_obj[idx].item())
                            except Exception:
                                sim_steps = 0
                        else:
                            sim_steps = int(steps_obj)
                    except Exception:
                        sim_steps = 0
                    # Per-env randomized phase and direction so starting angle and direction vary
                    # Re-randomize on first step after reset for each env (sim_steps == 0)
                    try:
                        if sim_steps == 0:
                            # Use seeded torch RNG for determinism across runs with the same --seed
                            rand_phase = float(torch.rand(1, device=self.device).item())  # [0,1)
                            self.sweep_phase_offsets[env_idx] = -math.pi + (2.0 * math.pi * rand_phase)
                            dir_flag = int(torch.randint(low=0, high=2, size=(1,), device=self.device).item())
                            self.sweep_directions[env_idx] = 1.0 if dir_flag == 1 else -1.0
                        phi0 = self.sweep_phase_offsets[env_idx]
                        direction = self.sweep_directions[env_idx]
                    except Exception:
                        phi0 = 0.0
                        direction = 1.0
                    angle_offset_degrees = A * math.sin(direction * (omega * sim_steps) + phi0)
                    debug_max_range = A
                    # [YawSweep DEBUG DISABLED]
                else:
                    # Spawn-aware angle selection: keep both gate (0°) and drone inside FOV; or 0 if disabled
                    # Extend the allowable angle range using the same per-level formula as yaw sweep:
                    # A = max(0, atan2(x_half, |y_center - base_y|) - 43.5) + 2.5 (deg)
                    # Also honor evaluation stretch during inference when enabled.
                    max_angle_range = _max_angle_range
                    try:
                        from aerial_gym.config.task_config.navigation_task_config_gate import task_config as _tc_fix
                        # Detect eval-stretch
                        try:
                            parent = getattr(self, 'env_manager', None)
                            gtd_local = parent.global_tensor_dict if (parent is not None and hasattr(parent, 'global_tensor_dict')) else {}
                        except Exception:
                            gtd_local = {}
                        eval_en = False
                        try:
                            eval_en = bool(gtd_local.get('eval_stretch_enabled', False))
                        except Exception:
                            eval_en = False
                        if not eval_en:
                            try:
                                import os as _os
                                eval_en = _os.environ.get("EVAL_STRETCH_ENABLED", "0").strip() in ("1", "true", "True")
                            except Exception:
                                eval_en = False
                        try:
                            eval_end = int(gtd_local.get('eval_stretch_end_level', getattr(_tc_fix.curriculum, 'eval_stretch_end_level', 23)))
                        except Exception:
                            eval_end = int(getattr(_tc_fix.curriculum, 'eval_stretch_end_level', 23))
                        eff_level = min(curriculum_level, eval_end) if eval_en else curriculum_level
                        sr_fix = _tc_fix.curriculum.get_spawn_ranges(eff_level)
                        x_half_fix = float(sr_fix.get('x_half_span_m', 0.5))
                        y_center_fix = float(sr_fix.get('y_center_m', -1.5))
                        dy_fix = abs(y_center_fix - float(base_y))
                        half_fov_fix = 87.0 * 0.5
                        margin_fix = 2.5
                        alpha_fix = math.degrees(math.atan2(x_half_fix, max(1e-6, dy_fix)))
                        sweep_like_max = max(0.0, alpha_fix - half_fov_fix) + margin_fix
                        max_angle_range = max(max_angle_range, sweep_like_max)
                    except Exception:
                        pass
                    # When sweep is disabled, honor the orientation randomization disable flag
                    if disable_flag or max_angle_range <= 0:
                        angle_offset_degrees = 0.0
                    else:
                        horizontal_fov = 87.0
                        half_fov = horizontal_fov * 0.5
                        margin = 2.5
                        if rp is not None and env_idx < rp.shape[0]:
                            cam_x, cam_y = base_camera_pos.x, base_camera_pos.y
                            dx = float(rp[env_idx, 0].item()) - cam_x
                            dy = float(rp[env_idx, 1].item()) - cam_y
                            theta_r = math.degrees(math.atan2(dx, dy))  # 0° points to +Y
                            gate_low, gate_high = -half_fov + margin, half_fov - margin
                            rob_low, rob_high = theta_r - (half_fov - margin), theta_r + (half_fov - margin)
                            low = max(gate_low, rob_low, -max_angle_range)
                            high = min(gate_high, rob_high, max_angle_range)
                            if high > low:
                                # Seeded torch RNG for deterministic selection
                                u = float(torch.rand(1, device=self.device).item())
                                angle_offset_degrees = low + u * (high - low)
                            else:
                                target = max(min(theta_r, gate_high), gate_low)
                                angle_offset_degrees = max(-max_angle_range, min(max_angle_range, target))
                        else:
                            # Seeded torch RNG for deterministic selection
                            u = float(torch.rand(1, device=self.device).item())
                            angle_offset_degrees = -max_angle_range + u * (2.0 * max_angle_range)
                        debug_max_range = max_angle_range
                
                # Store the angle for this environment
                if env_idx < len(self.current_camera_angles):
                    self.current_camera_angles[env_idx] = angle_offset_degrees
                
                # Convert to radians and update camera
                angle_offset_radians = angle_offset_degrees * (3.14159 / 180.0)
                # Euler jitter policy: avoid yaw jitter if curriculum yaw/sweep active. Apply only pitch (small tilt)
                jitter_roll_deg, jitter_pitch_deg, jitter_yaw_deg = self._euler_jitter_deg[env_idx] if (0 <= env_idx < len(self._euler_jitter_deg)) else (0.0, 0.0, 0.0)
                if sweep_enabled or (not disable_flag and max_angle_range > 0):
                    # Curriculum yaw active: zero yaw jitter
                    jitter_yaw_deg = 0.0
                # Apply pitch jitter as a small vertical target offset; roll is not supported via set_camera_location
                pitch_rad = jitter_pitch_deg * (3.14159 / 180.0)
                
                # Resolve per-env base Z (adaptive to gate center if requested)
                try:
                    env_base_z = base_z
                    if env_base_z is None:
                        gh = gtd.get('gate/center_height_per_env', None)
                        env_base_z = float(gh[env_idx].item()) if gh is not None else 1.5
                except Exception:
                    env_base_z = 1.5
                base_camera_env_pos = gymapi.Vec3(base_camera_pos.x, base_camera_pos.y, env_base_z)
                # Apply per-env translation jitter sampled at setup/reset
                try:
                    jx, jy, jz = self._trans_jitter[env_idx]
                except Exception:
                    jx, jy, jz = 0.0, 0.0, 0.0
                base_camera_env_pos = gymapi.Vec3(base_camera_env_pos.x + jx, base_camera_env_pos.y + jy, base_camera_env_pos.z + jz)

                # Calculate offset target position based on randomized angle for this environment
                target_distance = abs(base_camera_env_pos.y)  # Keep look-at distance consistent with base Y
                # Apply additional small yaw jitter around the curriculum yaw (if allowed)
                yaw_total = angle_offset_radians + (jitter_yaw_deg * (3.14159 / 180.0))
                target_x = base_camera_env_pos.x + target_distance * math.sin(yaw_total)
                target_y = base_camera_env_pos.y + target_distance * math.cos(yaw_total)
                # Look at gate adaptive center height, to keep camera pitched to the center
                try:
                    gh = gtd.get('gate/center_height_per_env', None)
                    target_z = float(gh[env_idx].item()) if gh is not None else env_base_z
                except Exception:
                    target_z = env_base_z
                # Apply pitch jitter as small vertical offset in look-at target
                target_z = target_z + math.tan(pitch_rad) * target_distance
                new_target = gymapi.Vec3(target_x, target_y, target_z)
                
                # Update ONLY this environment's camera
                env_handle = self.env_handles[env_idx]
                cam_handle = self.camera_handles[env_idx]
                self.gym.set_camera_location(cam_handle, env_handle, base_camera_env_pos, new_target)
                # Update debug caches
                try:
                    self.last_camera_pos[env_idx] = (float(base_camera_env_pos.x), float(base_camera_env_pos.y), float(base_camera_env_pos.z))
                    self.last_camera_target[env_idx] = (float(new_target.x), float(new_target.y), float(new_target.z))
                    self.last_angle_deg[env_idx] = float(angle_offset_degrees)
                except Exception:
                    pass
                # Debug only for env 0 to avoid spam
                if env_idx == 0:
                    # logger.warning(
                    #     f"[StaticCamReset] env0 base_y={base_camera_env_pos.y:.3f} env_base_z={base_camera_env_pos.z:.3f} "
                    #     f"pos=({base_camera_env_pos.x:.3f},{base_camera_env_pos.y:.3f},{base_camera_env_pos.z:.3f}) "
                    #     f"target=({new_target.x:.3f},{new_target.y:.3f},{new_target.z:.3f}) angle_deg={angle_offset_degrees:.1f}"
                    # )
                    pass
                
                # [YawSweep DEBUG DISABLED] logger.warning(f"[YawSweep] Updated static camera for env {env_idx} - Level {curriculum_level}: {angle_offset_degrees:.1f}° (max range: ±{debug_max_range:.1f}°)")
            
            # [YawSweep DEBUG DISABLED] logger.warning(f"[YawSweep] Updated static camera orientation for {len(env_ids)} environments")
            
        except Exception as e:
            logger.warning(f"Failed to update static camera orientation: {e}")
            # Fall back to fixed positioning if update fails
            logger.debug(f"Static camera orientation update failed - using fixed positioning")
            return
    
    def update_dynamic_camera_following(self, robot_positions, gate_positions, gate_center_heights):
        """Reimplemented dynamic follow: keep camera 1 m behind the drone (−Y),
        same height, and primarily look at the drone. If the gate is far outside
        the view, minimally steer the look target toward the gate while keeping
        the drone in frame.

        Args:
            robot_positions: (N,3) drone world positions
            gate_positions:  (N,3) gate world positions
            gate_center_heights: (N,) gate center Z per env
        """
        if hasattr(self, 'use_synthetic_camera') and self.use_synthetic_camera:
            return
        if not self.camera_setup_success or len(self.camera_handles) == 0:
            return
        try:
            from isaacgym import gymapi
            import math
            # Fixed offsets in world frame
            x_off, y_off, z_off = 0.0, -1.0, 0.0
            half_fov = 87.0 * 0.5
            margin = 5.0
            for env_idx in range(min(len(self.env_handles), len(self.camera_handles), robot_positions.shape[0])):
                drone = robot_positions[env_idx]
                cam_x = float(drone[0].item()) + x_off
                cam_y = float(drone[1].item()) + y_off
                cam_z = float(drone[2].item()) + z_off
                camera_pos = gymapi.Vec3(cam_x, cam_y, cam_z)

                # Default look target: the drone itself
                target_drone = gymapi.Vec3(float(drone[0].item()), float(drone[1].item()), float(drone[2].item()))

                # Check gate visibility and minimally bias toward it if necessary
                gate = gate_positions[env_idx]
                gate_cz = float(gate_center_heights[env_idx].item())
                # yaw to drone and gate (0° toward +Y)
                yaw_d = math.degrees(math.atan2(target_drone.x - cam_x, target_drone.y - cam_y))
                yaw_g = math.degrees(math.atan2(float(gate[0].item()) - cam_x, float(gate[1].item()) - cam_y))
                delta = yaw_g - yaw_d
                while delta > 180.0:
                    delta -= 360.0
                while delta < -180.0:
                    delta += 360.0
                # If gate is outside FOV when centered on drone, gently blend target
                if abs(delta) > (half_fov - margin):
                    w = 0.2  # small bias toward gate
                    tgx = (1.0 - w) * target_drone.x + w * float(gate[0].item())
                    tgy = (1.0 - w) * target_drone.y + w * float(gate[1].item())
                    tgz = (1.0 - w) * target_drone.z + w * gate_cz
                    camera_target = gymapi.Vec3(tgx, tgy, tgz)
                else:
                    camera_target = target_drone

                cam_handle = self.camera_handles[env_idx]
                env_handle = self.env_handles[env_idx]
                self.gym.set_camera_location(cam_handle, env_handle, camera_pos, camera_target)
        except Exception as e:
            logger.warning(f"Failed to update dynamic camera following: {e}")
            return

    def update_locked_follow(self, robot_positions):
        """Keep camera position fixed; rotate to always center the drone.

        Args:
            robot_positions: Tensor (num_envs, 3) with drone positions in world coords.
        """
        if hasattr(self, 'use_synthetic_camera') and self.use_synthetic_camera:
            return
        if not self.camera_setup_success or len(self.camera_handles) == 0:
            return
        try:
            # Base camera position for all envs
            from isaacgym import gymapi
            base_y = float(os.environ.get('SF_STATIC_CAMERA_BASE_Y', -3.0))
            base_z_env = os.environ.get('SF_STATIC_CAMERA_BASE_Z', '1.5')
            if isinstance(base_z_env, str) and base_z_env.strip().lower() == 'adaptive':
                # Try to read per-env adaptive Z from global tensors; fallback to 1.5
                try:
                    gtd = self.env_manager.IGE_env.global_tensor_dict
                    gate_center_per_env = gtd.get('gate/center_height_per_env', None)
                except Exception:
                    gate_center_per_env = None
            else:
                gate_center_per_env = None
            for env_idx in range(min(len(self.env_handles), len(self.camera_handles), robot_positions.shape[0])):
                cam_x = 0.0
                cam_y = base_y
                if gate_center_per_env is not None and env_idx < len(gate_center_per_env):
                    cam_z = float(gate_center_per_env[env_idx].item())
                else:
                    try:
                        cam_z = float(base_z_env)
                    except Exception:
                        cam_z = 1.5
                # Build camera and target
                camera_pos = gymapi.Vec3(cam_x, cam_y, cam_z)
                drone = robot_positions[env_idx]
                target = gymapi.Vec3(float(drone[0].item()), float(drone[1].item()), float(drone[2].item()))
                cam_handle = self.camera_handles[env_idx]
                env_handle = self.env_handles[env_idx]
                self.gym.set_camera_location(cam_handle, env_handle, camera_pos, target)
        except Exception as e:
            logger.warning(f"Failed to update locked-follow camera: {e}")
            return
    
    def capture_images(self, batched=False):
        """Capture depth and segmentation images from static camera.
        Args:
            batched: when True, return stacked depth of shape (num_envs, H, W) and seg for env0;
                     when False (default), return only env0 depth (H, W) and seg for compatibility
                     with GIF/debug pipelines.
        """
        if hasattr(self, 'use_synthetic_camera') and self.use_synthetic_camera:
            logger.error("Static camera synthetic mode disabled; no images will be generated.")
            return None, None
        
        if not self.camera_setup_success or len(self.camera_handles) == 0:
            # One-time lazy re-initialization attempt
            try:
                if (not hasattr(self, '_lazy_setup_attempted')) or (self._lazy_setup_attempted is False):
                    self._lazy_setup_attempted = True
                    logger.warning("Static camera not set up; attempting one-time lazy initialization")
                    try:
                        self._setup_static_camera()
                    except Exception as e:
                        logger.warning(f"Lazy static camera setup attempt failed: {e}")
                    # Re-check after lazy init
                    if not self.camera_setup_success or len(self.camera_handles) == 0:
                        logger.error("Static camera still not set up after lazy init; returning no images.")
                        return None, None
                else:
                    logger.error("Static camera not set up; returning no images.")
                    return None, None
            except Exception:
                logger.error("Static camera not set up; returning no images.")
                return None, None
        
        try:
            # Step graphics and render all cameras
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            
            if batched:
                # Get images from all environments so each env has its own view
                depth_imgs = []
                seg_imgs = []
                for i, (env_handle, cam_handle) in enumerate(zip(self.env_handles, self.camera_handles)):
                    # Depth image
                    depth_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH
                    )
                    depth_img = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
                    depth_imgs.append(depth_img)
                    # Segmentation image (used only for GIF/debug, keep env0 for compatibility)
                    seg_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env_handle, cam_handle, gymapi.IMAGE_SEGMENTATION
                    )
                    seg_img = gymtorch.wrap_tensor(seg_tensor).cpu().numpy()
                    seg_imgs.append(seg_img)
                
                # End access to image tensors
                self.gym.end_access_image_tensors(self.sim)
                
                # Stack per-env depth and normalize to [0, 1] like DCE expects
                if len(depth_imgs) > 0 and depth_imgs[0] is not None:
                    depth_stack = np.stack(depth_imgs, axis=0)
                    depth_stack[depth_stack == -np.inf] = 20.0  # Use far_plane value
                    depth_stack = np.abs(depth_stack)
                    depth_stack = np.clip(depth_stack, 0.4, 20.0)
                    depth_stack = (depth_stack - 0.4) / (20.0 - 0.4)
                    depth_stack = depth_stack.astype(np.float32)
                else:
                    depth_stack = None
                
                # Keep only env0 segmentation for downstream GIF/debug compatibility
                seg_img0 = seg_imgs[0] if len(seg_imgs) > 0 else None
                
                return depth_stack, seg_img0
            else:
                # Get images from camera 0 (single env for GIF/debug)
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
                
                # Normalize depth to [0,1]
                if depth_img is not None:
                    depth_normalized = depth_img.copy()
                    depth_normalized[depth_normalized == -np.inf] = 20.0
                    depth_normalized = np.abs(depth_normalized)
                    depth_normalized = np.clip(depth_normalized, 0.4, 20.0)
                    depth_normalized = (depth_normalized - 0.4) / (20.0 - 0.4)
                    depth_img = depth_normalized.astype(np.float32)
                
                return depth_img, seg_img
                
        except Exception as e:
            logger.error(f"Static camera capture error: {e}")
            return None, None
    def _generate_synthetic_camera_data(self):
        """Generate synthetic camera data for headless training."""
        try:
            # Create synthetic depth image (240x135) with reasonable gate-like features
            height, width = 135, 240
            depth_img = np.full((height, width), 0.5, dtype=np.float32)  # Mid-range depth
            
            # Add gate-like features to the synthetic depth (scaled to new resolution)
            gate_w = max(1, width // 4)   # ~60 px at 240 width
            gate_h = max(1, height // 3)  # ~45 px at 135 height
            gate_x_start = width // 2 - gate_w // 2
            gate_x_end = width // 2 + gate_w // 2
            gate_y_start = height // 2 - gate_h // 2
            gate_y_end = height // 2 + gate_h // 2
            
            # Gate opening (closer depth)
            depth_img[gate_y_start:gate_y_end, gate_x_start:gate_x_end] = 0.8
            
            # Gate frame (farther depth)
            frame_thickness = max(1, min(width, height) // 24)  # scale thickness
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
            return np.zeros((135, 240), dtype=np.float32), np.zeros((135, 240), dtype=np.uint8)


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
    boundary_violation_one_shot_mask,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
    
    # Base reward computation - REDUCED multiplication factor to prevent over-rewarding
    MULTIPLICATION_FACTOR_REWARD = 1.0 + (0.5) * curriculum_progress_fraction
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
    # Passage window covers the entire gate opening
    gate_passage_width_tolerance = gate_width * 0.5  # half-width
    gate_min_height = gate_position[:, 2] + gate_height * 0.0
    gate_max_height = gate_position[:, 2] + gate_height * 1.0
    
    just_passed_gate = (
        (robot_position[:, 1] > gate_position[:, 1]) &  # In front of gate
        (torch.abs(robot_position[:, 0] - gate_position[:, 0]) < gate_passage_width_tolerance) &  # Within gate width
        (robot_position[:, 2] > gate_min_height) & (robot_position[:, 2] < gate_max_height) &  # Within gate height
        (~gate_passed)  # Haven't passed before
    )
    
    # Center passage bonus: piecewise tiers by proximity to gate center (ADAPTIVE)
    gate_passage_reward = torch.zeros_like(gate_distance)
    gate_passage_reward[just_passed_gate] = parameter_dict["gate_passage_reward_magnitude"]
    
    gate_center_passage_bonus = torch.zeros_like(gate_distance)
    bonus_mag = parameter_dict["gate_center_passage_bonus_magnitude"]

    # Tolerances (as fractions of gate size) — denser piecewise
    x_tol_01  = gate_width  * 0.01
    x_tol_02  = gate_width  * 0.02
    x_tol_03  = gate_width  * 0.03
    x_tol_05  = gate_width  * 0.05
    x_tol_07  = gate_width  * 0.07
    x_tol_10  = gate_width  * 0.10
    x_tol_12  = gate_width  * 0.12
    x_tol_15  = gate_width  * 0.15
    x_tol_20  = gate_width  * 0.20
    z_tol_01  = gate_height * 0.01
    z_tol_02  = gate_height * 0.02
    z_tol_03  = gate_height * 0.03
    z_tol_05  = gate_height * 0.05
    z_tol_07  = gate_height * 0.07
    z_tol_10  = gate_height * 0.10
    z_tol_125 = gate_height * 0.125

    # Define tier masks (mutually exclusive, most strict first)
    t1 =  just_passed_gate & (x_distance_from_center < x_tol_01) & (z_distance_from_center < z_tol_01)
    t2 =  just_passed_gate & (~t1) & (x_distance_from_center < x_tol_02) & (z_distance_from_center < z_tol_02)
    t3 =  just_passed_gate & (~(t1 | t2)) & (x_distance_from_center < x_tol_03) & (z_distance_from_center < z_tol_03)
    t4 =  just_passed_gate & (~(t1 | t2 | t3)) & (x_distance_from_center < x_tol_05) & (z_distance_from_center < z_tol_05)
    t5 =  just_passed_gate & (~(t1 | t2 | t3 | t4)) & (x_distance_from_center < x_tol_07) & (z_distance_from_center < z_tol_07)
    t6 =  just_passed_gate & (~(t1 | t2 | t3 | t4 | t5)) & (x_distance_from_center < x_tol_10) & (z_distance_from_center < z_tol_10)
    t7 =  just_passed_gate & (~(t1 | t2 | t3 | t4 | t5 | t6)) & (x_distance_from_center < x_tol_12) & (z_distance_from_center < z_tol_10)
    t8 =  just_passed_gate & (~(t1 | t2 | t3 | t4 | t5 | t6 | t7)) & (x_distance_from_center < x_tol_15) & (z_distance_from_center < z_tol_125)
    t9 =  just_passed_gate & (~(t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8)) & (x_distance_from_center < x_tol_20) & (z_distance_from_center < z_tol_125)

    # Assign piecewise bonuses (fractions of bonus_mag)
    gate_center_passage_bonus[t1] = 1.00 * bonus_mag
    gate_center_passage_bonus[t2] = 0.92 * bonus_mag
    gate_center_passage_bonus[t3] = 0.85 * bonus_mag
    gate_center_passage_bonus[t4] = 0.77 * bonus_mag
    gate_center_passage_bonus[t5] = 0.65 * bonus_mag
    gate_center_passage_bonus[t6] = 0.55 * bonus_mag
    gate_center_passage_bonus[t7] = 0.45 * bonus_mag
    gate_center_passage_bonus[t8] = 0.35 * bonus_mag
    gate_center_passage_bonus[t9] = 0.25 * bonus_mag
    
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
    # Gate the camera-facing reward in TorchScript path as well: only before first crossing and while approaching gate
    pre_cross_approach = (robot_position[:, 1] < gate_position[:, 1] - 0.1) & (~gate_passed)
    gated_camera_facing = camera_facing_reward * pre_cross_approach.float()
    multiplied_camera_facing = MULTIPLICATION_FACTOR_REWARD * gated_camera_facing
    multiplied_altitude_maintenance = MULTIPLICATION_FACTOR_REWARD * altitude_maintenance_reward

    # Boundary violation penalty: one-shot mask computed in Python to avoid repeated penalties
    boundary_violation_penalty = torch.zeros_like(gate_distance)
    boundary_violation_penalty[boundary_violation_one_shot_mask] = -parameter_dict["boundary_violation_penalty_magnitude"]

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
        + multiplied_camera_facing  # Camera facing reward (gated pre-crossing)
        + multiplied_altitude_maintenance  # NEW: Altitude maintenance reward
        + boundary_violation_penalty  # Penalty for crossing behind gate plane without a valid passage
        + total_action_penalty
    )

    # Apply collision penalties
    reward[:] = torch.where(
        crashes > 0,
        parameter_dict["collision_penalty"] * torch.ones_like(reward),
        reward,
    )
    
    return reward, crashes, camera_gate_alignment