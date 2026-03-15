from __future__ import annotations

from aerial_gym.task.base_task import BaseTask, StepReturn
from aerial_gym.task.navigation_task_gate.debug_logging import NavigationTaskGateDebugMixin
from aerial_gym.task.navigation_task_gate.camera_observations import NavigationTaskGateCameraMixin
from aerial_gym.task.navigation_task_gate.curriculum_management import NavigationTaskGateCurriculumMixin
from aerial_gym.task.navigation_task_gate.gate_geometry import NavigationTaskGateGeometryMixin
from aerial_gym.task.task_config_protocol import TaskConfig
from aerial_gym.task.schemas import (
    GATE_OBS_LAYOUT,
    EpisodeRewardAccumulators,
    EpisodeTrajectoryState,
)
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

from aerial_gym.sensors.static_camera_manager import StaticCameraManager
from aerial_gym.utils.env_flag_utils import (
    read_env_bool,
    read_env_int,
    parse_ablation_flags,
    apply_ablation_flags_to_tensor_dict,
)
from aerial_gym.task.navigation_task_gate.reward_functions import compute_gate_reward

logger = CustomLogger("navigation_task_gate")


def dict_to_class(dict) -> None:
    return type("ClassFromDict", (object,), dict)


class NavigationTaskGate(NavigationTaskGateGeometryMixin, NavigationTaskGateCurriculumMixin, NavigationTaskGateCameraMixin, NavigationTaskGateDebugMixin, BaseTask):
    def __init__(
        self,
        task_config: TaskConfig,
        seed: int | None = None,
        num_envs: int | None = None,
        headless: bool | None = None,
        device: str | None = None,
        use_warp: bool | None = None,
    ) -> None:
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
                        except (ValueError, TypeError):
                            continue
                        if rhs in ('zero', 'zerograd') and a <= 86 and b >= 150:
                            static_ablated = True
                            break
            if static_ablated:
                self.task_config.reward_parameters["static_fov_visibility_reward_magnitude"] = 0.0
        except (ValueError, TypeError):
            pass

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

        # This ensures the asset manager gets the correct count from the start
        self.curriculum_level = self.task_config.curriculum.min_level
        # Obstacle ablation: if disabled, force to fixed count (default 0)
        obstacles_disable = read_env_bool(
            "SF_DISABLE_OBSTACLE_RANDOMIZATION", self.task_config.disable_obstacle_randomization
        )
        obstacles_fixed = read_env_int(
            "SF_FIXED_OBSTACLES_BEHIND_GATE", self.task_config.fixed_obstacles_behind_gate
        )
        if obstacles_disable:
            obstacles_behind_gate = max(0, obstacles_fixed)
        else:
            obstacles_behind_gate = self.task_config.curriculum.get_obstacle_count_behind_gate(self.curriculum_level)
        
        # NOTE: Even though 11 gate variants are loaded, only 1 will be visible at any time
        # The other 10 gates are hidden by moving them to (-1000, -1000, -1000)
        visible_gates = 0  # Only 1 gate visible at a time (others hidden by gate selection system)
        walls = 6  # 6 boundary walls
        robot = 0  # Robot is NOT part of env_asset_state_tensor (handled separately)
        fixed_assets_visible = visible_gates + walls  # = 7 visible fixed assets
        total_obstacles_in_env = fixed_assets_visible + obstacles_behind_gate
        
        logger.info(f"PRE-INIT: Setting curriculum level {self.curriculum_level} with {obstacles_behind_gate} curriculum obstacles")
        logger.info(f"PRE-INIT: Visible assets (env assets only): {visible_gates} gate + {walls} walls + {obstacles_behind_gate} curriculum = {total_obstacles_in_env} total")
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
        
        # Propagate ablation flags to global tensor dict for EnvManager to consume
        if hasattr(self.sim_env, "global_tensor_dict"):
            flags = parse_ablation_flags(self.task_config)
            apply_ablation_flags_to_tensor_dict(
                self.sim_env.global_tensor_dict, flags, self.task_config, logger
            )
            # Mirror commonly-accessed flags onto self for quick access
            self.disable_static_camera_orientation_randomization = flags[
                "static_camera_orient_disabled"
            ]
            self.disable_camera_frame_dropout_randomization = flags[
                "camera_frame_dropout_disabled"
            ]
            self.disable_camera_noise_randomization = flags["camera_noise_disabled"]
            self.disable_state_noise_randomization = flags["state_noise_disabled"]
            self.disable_dynamic_camera_following = flags[
                "dynamic_camera_following_disabled"
            ]

        # Immediately select a random gate variant once after creation (safety)
        if hasattr(self.sim_env, 'apply_gate_variant_selection'):
            logger.warning("[GateVariant] Initial selection after build (one-time)")
            self.sim_env.apply_gate_variant_selection(env_ids=torch.arange(self.sim_env.num_envs, device=self.device))
        
        if hasattr(self.sim_env, 'global_tensor_dict'):
            # Override count if obstacle randomization disabled
            try:
                obs_dis = bool(self.sim_env.global_tensor_dict.get('obstacles_randomization/disabled', False))
            except (KeyError, TypeError):
                obs_dis = False
            if obs_dis:
                try:
                    fixed_count = int(self.sim_env.global_tensor_dict.get('obstacles_randomization/fixed_count', 0))
                except (ValueError, TypeError):
                    fixed_count = 0
                total_obstacles_in_env = fixed_assets_visible + max(0, fixed_count)
            self.sim_env.global_tensor_dict["num_obstacles_in_env"] = total_obstacles_in_env
            logger.info(f"POST-INIT: Updated global_tensor_dict with obstacle count: {total_obstacles_in_env}")

        # Target position: will be set to adaptive gate center on each reset
        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        # Target position is now always set to adaptive gate center instead of random sampling

        self.success_aggregate = 0
        self.crashes_aggregate = 0
        self.timeouts_aggregate = 0

        self._init_gate_tracking_tensors()
        self._init_vae_model()

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
            if not True:
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
            if'observations' in self.task_obs:
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
                
        except RuntimeError as e:
            logger.warning(f"❌ Error in process_obs_for_task(): {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
        
        logger.warning("="*80)
        
        self._init_curriculum()

        # Use dedicated terminations tensor if provided by env_manager; fallback to crashes
        try:
            self.terminations = self.obs_dict["terminations"]
        except (KeyError, TypeError):
            self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        self._init_observation_action_spaces()
        self._init_task_observations()

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
        
        self._init_episode_reward_tracking()
        self._init_episode_trajectory_state()
        self._init_debug_flags()

        # Initialize gate dimensions for all environments after full initialization
        logger.warning("[GATE_ADAPTIVE] Initializing gate dimensions for all environments")
        self.update_gate_dimensions_for_environments(torch.arange(self.sim_env.num_envs, device=self.device))

        # Ensure infos survive resets for logging back to the learner
        self._infos_to_return = None

    # Private init helpers (extracted from __init__ for readability)

    def _init_curriculum(self) -> None:
        """Initialize curriculum level, obstacle counts, camera difficulty, and logging."""
        # Use the curriculum level that was already set during pre-initialization
        # Force curriculum level if requested via env/task config
        forced = os.environ.get('SF_FORCE_CURRICULUM_LEVEL', None)
        if forced is None:
            forced = self.task_config.force_curriculum_level
        if forced is not None:
            self.curriculum_level = int(forced)
        self.obs_dict["curriculum_level"] = self.curriculum_level
        
        # Track maximum curriculum level reached (for no-decrease policy)
        self.max_curriculum_level_reached = self.curriculum_level
        
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
        
        logger.info(f"INITIAL CURRICULUM (Level {self.curriculum_level}):")
        logger.info(f"   1. OBSTACLES: {obstacles_behind_gate} behind gate (total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)")
        try:
            # Determine baseline level and ablation flags
            baseline_level = int(self.task_config.curriculum.min_level)
            pos_dis = False; yaw_dis = False
            gtd = self.sim_env.global_tensor_dict
            pos_dis = bool(gtd.get('spawn_randomization/position_disabled', False))
            yaw_dis = bool(gtd.get('spawn_randomization/orientation_disabled', False))
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
        except (ValueError, TypeError) as e:
            logger.info(f"   2. SPAWN: (fallback) Using fixed LMF2 config due to: {e}")
        # 3. STATIC CAMERA YAW SWEEP STATUS (takes precedence over static orientation randomization)
        try:
            yaw_enabled = str(os.environ.get('SF_ENABLE_STATIC_CAMERA_YAW_SWEEP', 'false')).lower() == 'true'
            yaw_speed = float(os.environ.get('SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG', '10.0'))
        except (ValueError, TypeError):
            yaw_enabled = False
            yaw_speed = 10.0
        # Orientation randomization disable flag and dynamic camera effective state
        try:
            cam_orient_disabled = bool(self.sim_env.global_tensor_dict.get('static_camera_randomization/orientation_disabled', False))
        except (KeyError, TypeError):
            cam_orient_disabled = False
        try:
            dyn_cfg = self.task_config.curriculum.enable_dynamic_camera_following
            dyn_dis = bool(self.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False))
            dynamic_effective = bool(dyn_cfg and not dyn_dis)
        except (KeyError, TypeError):
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
        except (ValueError, TypeError):
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
        except (KeyError, TypeError):
            state_noise_disabled = bool(self.disable_state_noise_randomization)
        if self.task_config.curriculum.enable_state_noise and not state_noise_disabled:
            sn = self.task_config.curriculum.get_state_noise(self.curriculum_level)
            logger.info(
                f"   7. STATE NOISE: drone_pos_std={sn['drone_pos_std_m']:.4f} m, drone_orient_std={sn['drone_orient_std_rad']*57.2958:.3f} deg, "
                f"static_pos_std={sn['static_pos_std_m']:.4f} m, static_orient_std={sn['static_orient_std_rad']*57.2958:.3f} deg"
            )
        else:
            logger.info("   7. STATE NOISE: disabled")
        
        logger.info(f"   8. ASSET MANAGER: Updated both obs_dict and global_tensor_dict with count {total_obstacles_in_env}")
        # Curriculum multiplier debug (initial) - compute fraction directly (attribute may not exist yet)
        cm_disabled = read_env_bool("SF_DISABLE_CURRICULUM_MULTIPLIER", self.task_config.disable_curriculum_multiplier)
        if not cm_disabled:
            cm_disabled = bool(self.task_config.disable_curriculum_multiplier)
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


    def _init_gate_tracking_tensors(self) -> None:
        """Initialise gate-specific tracking tensors and adaptive dimensions."""
        self.gate_position = torch.zeros((self.sim_env.num_envs, 3), device=self.device)
        self.gate_approach_distance = torch.zeros(self.sim_env.num_envs, device=self.device)
        self.gate_width = torch.zeros((self.sim_env.num_envs,), device=self.device)
        self.gate_height = torch.zeros((self.sim_env.num_envs,), device=self.device)
        self.gate_center_height = torch.zeros((self.sim_env.num_envs,), device=self.device)
        self.gate_scale_factors = torch.ones((self.sim_env.num_envs,), device=self.device)

    def _init_vae_model(self) -> None:
        """Set up shared VAE encoder (or identity fallback) and latent buffers."""
        if self.task_config.vae_config.use_vae:
            self.shared_vae_model = VAEImageEncoder(
                config=self.task_config.vae_config, device=self.device
            )
            latent_dims = self.task_config.vae_config.latent_dims
            self.image_latents = torch.zeros(
                (self.sim_env.num_envs, latent_dims),
                device=self.device,
                requires_grad=False,
            )
            self.static_image_latents = torch.zeros(
                (self.sim_env.num_envs, latent_dims),
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

    def _init_observation_action_spaces(self) -> None:
        """Define Gymnasium observation and action spaces."""
        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.task_config.observation_space_dim,),
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
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_transformation_function = self.task_config.action_transformation_function
        self.num_envs = self.sim_env.num_envs

    def _init_task_observations(self) -> None:
        """Allocate task observation tensors."""
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

    def _init_episode_reward_tracking(self) -> None:
        """Allocate per-component episode reward accumulators and statistics tensors."""
        self.episode_rewards = EpisodeRewardAccumulators.create(self.num_envs, self.device)
        # Backward-compatible aliases (used by update_episode_reward_tracking, reset, logging)
        self.episode_pos_reward = self.episode_rewards.pos_reward
        self.episode_very_close_reward = self.episode_rewards.very_close_reward
        self.episode_getting_closer_reward = self.episode_rewards.getting_closer_reward
        self.episode_gate_approach_reward = self.episode_rewards.gate_approach_reward
        self.episode_gate_alignment_reward = self.episode_rewards.gate_alignment_reward
        self.episode_camera_facing_reward = self.episode_rewards.camera_facing_reward
        self.episode_action_penalty = self.episode_rewards.action_penalty
        self.episode_gate_passage_reward = self.episode_rewards.gate_passage_reward
        self.episode_collision_penalty = self.episode_rewards.collision_penalty
        self.episode_image_reward = self.episode_rewards.image_reward
        self.episode_static_fov_visibility_reward = self.episode_rewards.static_fov_visibility_reward
        self.episode_boundary_violation_penalty = self.episode_rewards.boundary_violation_penalty
        self.episode_lengths = self.episode_rewards.lengths
        self.episode_time_penalty = self.episode_rewards.time_penalty
        self.episode_timeout_penalty = self.episode_rewards.timeout_penalty
        self.completed_episodes: list[dict[str, float]] = []
        self.max_stored_episodes = 10

    def _init_episode_trajectory_state(self) -> None:
        """Allocate per-environment episode trajectory tracking state."""
        self.trajectory = EpisodeTrajectoryState.create(self.num_envs, self.device)
        # Backward-compatible aliases (used throughout step/reset/curriculum code)
        self._episode_fresh = self.trajectory.fresh
        self._ep_spawn_pos = self.trajectory.spawn_pos
        self._ep_gate_center_at_spawn = self.trajectory.gate_center_at_spawn
        self._ep_last_pos = self.trajectory.last_pos
        self._ep_path_len = self.trajectory.path_len
        self._ep_steps = self.trajectory.steps
        self._ep_min_gate_dist = self.trajectory.min_gate_dist
        self._ep_gate_crossed = self.trajectory.gate_crossed
        self._ep_time_to_gate = self.trajectory.time_to_gate
        self._ep_center_offset_cross = self.trajectory.center_offset_cross
        self._ep_height_offset_cross = self.trajectory.height_offset_cross
        self._bv_flag_episode = self.trajectory.bv_flag_episode
        self._bv_prev_env0 = False

    def _init_debug_flags(self) -> None:
        """Initialize one-shot debug/logging flags so hasattr checks are never needed."""
        self._drone_cam_debug_last = False
        self._drone_vae_debug_last = False
        self._static_cam_debug_last = False
        self._static_cam_depth_logged = False
        self._static_debug_logged = False
        self._vae_output_logged = False
        self._vae_debug_logged = False
        self._no_static_logged = False
        self._return_tuple_debug_printed = False
        self._final_verification_printed = False
        self._prev_drone_depth = None
        self._prev_static_depth = None
        self._success_window_history = []
        self._curriculum_cooldown = 0
        self.min_pixel_dist = None
        self.curriculum_log_file = None
        self._curriculum_multiplier_factor = 1.0
        self._debug_cam_world = None
        self._debug_drone_eul = None
        self._debug_cam_eul = None
        self.max_curriculum_level_reached = self.curriculum_level
        self._last_traj_metrics_per_env = {}
        self._last_traj_metrics_avg = {}

    def logging_sanity_check(self, infos: dict[str, torch.Tensor]) -> None:
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

    def close(self) -> None:
        try:
            if hasattr(self.sim_env, 'delete_env'):
                self.sim_env.delete_env()
            elif hasattr(self.sim_env, 'close'):
                self.sim_env.close()
            else:
                print("[DEBUG] No cleanup method found for sim_env")
        except (ValueError, TypeError) as e:
            print(f"[DEBUG] Error during close: {e}")
        
        # Close curriculum log file
        if self.curriculum_log_file:
            try:
                import datetime
                self.curriculum_log_file.write(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training session ended.\n")
                self.curriculum_log_file.write("="*80 + "\n")
                self.curriculum_log_file.close()
            except OSError as e:
                print(f"[DEBUG] Error closing curriculum log: {e}")



    def reset(self) -> StepReturn:
        self.reset_idx(torch.arange(self.sim_env.num_envs))
        return self.get_return_tuple()

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """
        SIMPLIFIED RESET WITH FIXED SPAWNING PARAMETERS
        
        Uses LMF2 robot config for spawning with fixed parameters:
        - ±0.5m lateral variation from gate center
        - ±45° orientation randomization  
        - Minimal initial velocity for randomization
        """
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
        self._ep_target_success_flag[env_ids] = False
        self.gate_approach_distance[env_ids] = 0.0
        
        # RESET EPISODE REWARD TRACKING: Store completed episode data and reset trackers
        self.reset_episode_reward_tracking(env_ids)
        
        # Update static camera position based on curriculum level (ONLY for resetting environments)
        if len(env_ids) > 0:
            self.static_camera_manager.update_camera_positions(self.curriculum_level, env_ids)
            logger.debug(f"Updated static camera angles for {len(env_ids)} resetting environments: {env_ids.tolist()}")
        
        # Add debugging for gate randomization on reset
        
        # Persist curriculum level for env manager (for gated gate randomization)
        self.sim_env.global_tensor_dict["curriculum_level"] = int(self.curriculum_level)
        # Propagate eval-stretch flags from task_config if present (for gate size unlocking etc.)
        stretch_enabled = bool(self.task_config.curriculum.eval_stretch_enabled)
        stretch_end = int(self.task_config.curriculum.eval_stretch_end_level)
        self.sim_env.global_tensor_dict['eval_stretch_enabled'] = stretch_enabled
        self.sim_env.global_tensor_dict['eval_stretch_end_level'] = stretch_end
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
        self.trajectory.reset_envs(env_ids)
        
        self.infos = {}
        return
    
    
    
    
    # These methods have been removed as we now use fixed parameters from LMF2 config
    # The normal Isaac Gym reset mechanism handles spawning using min_init_state/max_init_state

    def render(self) -> None:
        return self.sim_env.render()
    def _update_trajectory_state(
        self, robot_position: torch.Tensor, gate_center_position: torch.Tensor, gate_passed_current: torch.Tensor
    ) -> None:
        """Update per-env episode trajectory tracking: spawn capture, path length, gate crossing."""
        fresh_mask = self._episode_fresh
        if torch.any(fresh_mask):
            self._ep_spawn_pos[fresh_mask] = robot_position[fresh_mask]
            _gcenter = self.gate_position.clone()
            _gcenter[:, 2] = _gcenter[:, 2] + self.gate_center_height
            self._ep_gate_center_at_spawn[fresh_mask] = _gcenter[fresh_mask]
            self._ep_last_pos[fresh_mask] = robot_position[fresh_mask]
            self._episode_fresh[fresh_mask] = False

        step_deltas = robot_position - self._ep_last_pos
        step_dist = torch.norm(step_deltas, dim=1)
        self._ep_path_len += step_dist
        self._ep_last_pos = robot_position
        self._ep_steps += 1

        step_gate_dist = torch.norm(robot_position - gate_center_position, dim=1)
        self._ep_min_gate_dist = torch.minimum(self._ep_min_gate_dist, step_gate_dist)

        newly_crossed = (~self._ep_gate_crossed) & gate_passed_current
        if torch.any(newly_crossed):
            self._ep_gate_crossed[newly_crossed] = True
            self._ep_time_to_gate[newly_crossed] = self._ep_steps[newly_crossed].to(torch.float32)
            dx_cross = robot_position[newly_crossed, 0] - gate_center_position[newly_crossed, 0]
            dz_cross = robot_position[newly_crossed, 2] - gate_center_position[newly_crossed, 2]
            self._ep_center_offset_cross[newly_crossed] = torch.sqrt(dx_cross * dx_cross + dz_cross * dz_cross)
            self._ep_height_offset_cross[newly_crossed] = torch.abs(dz_cross)

    def _validate_and_step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform actions, run physics step, detect NaN/Inf in actions and observations."""
        transformed_action = self.action_transformation_function(actions)
        # Action NaN/Inf guard
        try:
            invalid_action_mask = torch.any(torch.isnan(transformed_action) | torch.isinf(transformed_action), dim=1)
            if torch.any(invalid_action_mask):
                transformed_action[invalid_action_mask] = 0.0
                nan_trunc_mask = invalid_action_mask.clone()
                if self.task_config.guard_debug_enabled:
                    _ids = torch.nonzero(invalid_action_mask, as_tuple=False).squeeze(-1).tolist()
                    logger.warning(f"[NaNGuard] Invalid ACTION in envs {_ids}; zeroed and will truncate")
            else:
                nan_trunc_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        except RuntimeError:
            nan_trunc_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.sim_env.step(actions=transformed_action)

        # Observation NaN/Inf guard
        for k, v in self.obs_dict.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == self.num_envs:
                bad = torch.any(torch.isnan(v) | torch.isinf(v), dim=tuple(range(1, v.ndim)))
                if self.task_config.guard_debug_enabled and torch.any(bad):
                    _ids = torch.nonzero(bad, as_tuple=False).squeeze(-1).tolist()
                    logger.warning(f"[NaNGuard] Invalid OBS '{k}' in envs {_ids}")
                nan_trunc_mask |= bad
        if torch.any(nan_trunc_mask):
            if "robot_position" in self.obs_dict and isinstance(self.obs_dict["robot_position"], torch.Tensor):
                self.obs_dict["robot_position"][nan_trunc_mask] = 0.0

        return transformed_action, nan_trunc_mask

    def step(self, actions: torch.Tensor) -> StepReturn:
        # VELOCITY CONTROLLER: Transform 4D actions to direct velocity commands for LMF2 robot
        # Input: [x_vel_cmd, y_vel_cmd, z_vel_cmd, yaw_rate_cmd] ∈ [-1, 1]^4
        # Output: [x_vel, y_vel, z_vel, yaw_rate] applied directly as velocity commands
        
        transformed_action, nan_trunc_mask = self._validate_and_step(actions)

        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        self.rewards[:], self.terminations[:], camera_gate_alignment = self.compute_rewards_and_crashes(self.obs_dict)
        # Reward NaN/Inf guard: sanitize invalid rewards and truncate offending envs
        invalid_reward_mask = torch.isnan(self.rewards) | torch.isinf(self.rewards)
        if torch.any(invalid_reward_mask):
            if self.task_config.guard_debug_enabled:
                _ids = torch.nonzero(invalid_reward_mask, as_tuple=False).squeeze(-1).tolist()
                logger.warning(f"[NaNGuard] Invalid REWARD in envs {_ids}; zeroed and truncating.")
            self.rewards[invalid_reward_mask] = 0.0
            # Ensure truncation to reset these envs safely
            self.truncations[invalid_reward_mask] = 1


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
            if self.task_config.guard_debug_enabled:
                _ids = torch.nonzero(nan_trunc_mask, as_tuple=False).squeeze(-1).tolist()
                logger.warning(f"[NaNGuard] Truncating envs due to NaN/Inf: {_ids}")

        # Success = simply passing through the gate boundary (any part of the gate opening)
        # More forgiving than target-based or centered passage requirements
        robot_position = self.obs_dict["robot_position"]
        # Snapshot positions BEFORE any potential resets mutate the shared tensors
        robot_position_before_reset = robot_position.clone()
        
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
            self._ep_target_success_flag[immediate_success_mask] |= target_success_immediate[immediate_success_mask]
            try:
                success_envs = torch.nonzero(immediate_success_mask, as_tuple=False).squeeze(-1).tolist()
            except RuntimeError:
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
        end_success_mask = (self.terminations > 0) & (target_success & gate_passage_success) & (~crash_mask)
        self._ep_target_success_flag[end_success_mask] = True
        
        # Training success remains gate passage only; target success is for logging metrics only


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
        except (ValueError, TypeError):
            timeout_penalty = 75.0
        if torch.any(timeouts):
            # Apply to the per-env reward vector maintained at the task level
            self.rewards = self.rewards - (timeouts.float() * timeout_penalty)
            self.episode_timeout_penalty[timeouts] -= timeout_penalty
        
        # Add gate navigation specific info to wandb tracking
        # Calculate gate navigation metrics from current state
        robot_position = self.obs_dict["robot_position"]
        # Use geometric center of gate opening (z + center_height) so a perfect center pass can approach 0
        gate_center_position = self.gate_position.clone()
        gate_center_position[:, 2] = gate_center_position[:, 2] + self.gate_center_height
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
        
        # Skip logging of continuous curriculum/current_* to W&B entirely
        # (retain internal counters elsewhere if needed)

        # Update per-env episode trajectory state
        self._update_trajectory_state(robot_position, gate_center_position, gate_passed_current)

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
                except (ValueError, TypeError):
                    overall_success_rate = torch.tensor(float('nan'), device=self.device)
                    target_success_rate = torch.tensor(float('nan'), device=self.device)
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
                except (ValueError, TypeError):
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
                    # Build metrics dict while avoiding undefined time-to-gate when no crossing occurred
                    _metrics_avg = {
                        'path_efficiency': float(pe_avg.item()),
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
                    # Only include time-to-gate (steps/seconds) if any env in this reset batch actually crossed
                    try:
                        num_crossed = int(torch.isfinite(time_to_gate[env_ids]).sum().item())
                    except (ValueError, TypeError):
                        num_crossed = 0
                    if num_crossed > 0 and not torch.isnan(ttg_avg):
                        _metrics_avg['time_to_gate_steps'] = float(ttg_avg.item())
                        _metrics_avg['time_to_gate'] = float(ttg_avg.item())
                    self._last_traj_metrics_avg = _metrics_avg
                except (ValueError, TypeError):
                    self._last_traj_metrics_avg = None
                # Provide averaged metrics to infos['episode_extra_stats'] so learner can push to W&B as a backup
                extra = self.infos.get('episode_extra_stats', {})
                if not isinstance(extra, dict):
                    extra = {}
                extra.update(self._last_traj_metrics_avg or {})
                # Expose per-camera noise/frame-drop overrides to W&B, mirroring prior style
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
                self.infos['episode_extra_stats'] = extra
            except (ValueError, TypeError) as e:
                logger.debug(f"Trajectory metrics computation failed: {e}")
            # Stash infos to return to the learner before we clear them in reset
            self._infos_to_return = dict(self.infos)
            # Finally, reset environments and mark them fresh for next episode
            self.reset_idx(reset_envs)
        self.num_task_steps += 1
        # do stuff with the image observations here
        self.process_image_observation()
        self.process_static_camera_observation()
        self.post_image_reward_addition()
        
        # FINAL VERIFICATION: After all processing is complete
        if not self._final_verification_printed:
            self._final_verification_printed = True
            logger.warning("🎯 FINAL STATIC CAMERA VERIFICATION (AFTER PROCESSING):")
            
            # Process observations to get final state
            self.process_obs_for_task()
            
            if'observations' in self.task_obs:
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


    def post_image_reward_addition(self) -> None:
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
        if self.num_task_steps % 200 == 0:
            avg_min_dist = torch.mean(self.min_pixel_dist).item()
            avg_image_reward = torch.mean(image_rewards).item() if len(image_rewards) > 0 else 0.0
            min_pixel_dist = torch.min(self.min_pixel_dist).item()
            max_pixel_dist = torch.max(self.min_pixel_dist).item()
            
            # Count environments with different distance ranges
            very_close_count = torch.sum(self.min_pixel_dist < 2.0).item()  # < 2m
            close_count = torch.sum((self.min_pixel_dist >= 2.0) & (self.min_pixel_dist < 4.0)).item()  # 2-4m
            safe_count = torch.sum(self.min_pixel_dist >= 4.0).item()  # > 4m
            
        # Apply the image rewards
        self.rewards[~self.terminations] += image_rewards


    def get_return_tuple(self) -> StepReturn:
        self.process_obs_for_task()
        # If we have stashed infos from the previous step (pre-reset), use them once
        if self._infos_to_return is not None:
            infos_to_return = self._infos_to_return
            self._infos_to_return = None
        else:
            infos_to_return = self.infos
        
        # ADDITIONAL DEBUG: Verify observations in get_return_tuple (called every step)
        if not self._return_tuple_debug_printed:
            self._return_tuple_debug_printed = True
            logger.warning("🎯 OBSERVATION VERIFICATION IN get_return_tuple():")
            
            if'observations' in self.task_obs:
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
        if(int(self.num_task_steps) % 120 == 0):
            if'observations' in self.task_obs:
                obs_all = self.task_obs['observations']
                if obs_all.shape[0] > 0 and obs_all.shape[1] >= 9:
                    sp = obs_all[0, 3:6]
                    so = obs_all[0, 6:9]
                    # Drone and camera world positions
                    rp = self.obs_dict.get('robot_position', None)
                    rp0 = rp[0] if isinstance(rp, torch.Tensor) and rp.shape[0] > 0 else torch.zeros(3, device=self.device)
                    cw = self._debug_cam_world if self._debug_cam_world is not None else None
                    cw0 = cw[0] if isinstance(cw, torch.Tensor) and cw.shape[0] > 0 else torch.zeros(3, device=self.device)
                    # Drone & camera world orientations (Euler)
                    de = self._debug_drone_eul if self._debug_drone_eul is not None else None
                    ce = self._debug_cam_eul if self._debug_cam_eul is not None else None
                    de0 = de[0] if isinstance(de, torch.Tensor) and de.shape[0] > 0 else torch.zeros(3, device=self.device)
                    ce0 = ce[0] if isinstance(ce, torch.Tensor) and ce.shape[0] > 0 else torch.zeros(3, device=self.device)
        
        # Update camera modes in priority order: arc-follow (new mode) > dynamic-follow > yaw-sweep/locked-follow
        dynamic_enabled = self.task_config.curriculum.enable_dynamic_camera_following
        dynamic_disabled = bool(self.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False))
        arc_follow_enabled = bool(self.sim_env.global_tensor_dict.get('static_camera/arc_follow_enabled', False))
        
        if arc_follow_enabled:
            self.static_camera_manager.update_arc_follow(
                self.obs_dict["robot_position"],
                self.gate_position,
                self.gate_center_height,
                float(self.sim_env.global_tensor_dict.get('static_camera/arc_follow_radius_m', 2.0))
            )
        elif dynamic_enabled and not dynamic_disabled:
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
            if locked_follow and not (dynamic_enabled and not dynamic_disabled) and not arc_follow_enabled:
                # Update orientation-only to keep the drone centered
                self.static_camera_manager.update_locked_follow(self.obs_dict["robot_position"]) 
            elif sweep_enabled_flag and not (dynamic_enabled and not dynamic_disabled) and not arc_follow_enabled:
                env_ids_all = torch.arange(self.sim_env.num_envs, device=self.device)
                self.static_camera_manager.update_camera_positions(self.curriculum_level, env_ids_all)
                # Temporary debug for yaw sweep status (print ~once every 5 seconds at 60Hz)
                try:
                    sim_steps = int(gtd.get('sim_steps', torch.tensor([0], device=self.device))[0].item())
                except (ValueError, TypeError):
                    sim_steps = 0
                # Reduce spam: every 300 steps instead of every 60
                if (sim_steps % 300) == 0:
                    logger.warning("[YawSweep] Camera orientation update running (sweeping/locked-follow active)")
        except (ValueError, TypeError) as e:
            logger.debug(f"[YawSweep] Per-step update skipped due to: {e}")

        self._compute_visibility_metrics(infos_to_return)

        
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            infos_to_return,
        )


    def process_obs_for_task(self) -> None:
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
        # This provides the agent with complete spatial awareness of its state and static camera relative position
        
        drone_pos_clean = self.obs_dict["robot_position"]
        # Apply curriculum-driven state noise (drone position)
        if self.task_config.curriculum.enable_state_noise and not bool(self.sim_env.global_tensor_dict.get('state_randomization/noise_disabled', False)):
            noise_cfg = self.task_config.curriculum.get_state_noise(self.curriculum_level)
            dp_std = float(noise_cfg.get("drone_pos_std_m", 0.0))
            if dp_std > 0.0:
                drone_pos_noised = drone_pos_clean + torch.randn_like(drone_pos_clean) * dp_std
            else:
                drone_pos_noised = drone_pos_clean
        else:
            drone_pos_noised = drone_pos_clean
        obs = self.task_obs["observations"]
        layout = GATE_OBS_LAYOUT

        obs[:, layout.drone_position] = drone_pos_noised

        # Static camera pose (relative to drone)
        static_camera_pos, static_camera_orientation = self._get_static_camera_pose_relative_to_drone()
        if self.task_config.curriculum.enable_state_noise and not bool(self.sim_env.global_tensor_dict.get('state_randomization/noise_disabled', False)):
            noise_cfg = self.task_config.curriculum.get_state_noise(self.curriculum_level)
            sp_std = float(noise_cfg.get("static_pos_std_m", 0.0))
            so_std = float(noise_cfg.get("static_orient_std_rad", 0.0))
            if sp_std > 0.0:
                static_camera_pos = static_camera_pos + torch.randn_like(static_camera_pos) * sp_std
            if so_std > 0.0:
                static_camera_orientation = static_camera_orientation + torch.randn_like(static_camera_orientation) * so_std
                static_camera_orientation = torch.atan2(torch.sin(static_camera_orientation), torch.cos(static_camera_orientation))

        obs[:, layout.static_camera_position] = static_camera_pos
        obs[:, layout.static_camera_orientation] = static_camera_orientation

        # Drone orientation (roll, pitch, yaw)
        euler_angles = ssa(get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"]))
        if self.task_config.curriculum.enable_state_noise and not bool(self.sim_env.global_tensor_dict.get('state_randomization/noise_disabled', False)):
            noise_cfg = self.task_config.curriculum.get_state_noise(self.curriculum_level)
            do_std = float(noise_cfg.get("drone_orient_std_rad", 0.0))
            if do_std > 0.0:
                euler_angles = euler_angles + torch.randn_like(euler_angles) * do_std
                euler_angles = torch.atan2(torch.sin(euler_angles), torch.cos(euler_angles))
        obs[:, layout.drone_orientation] = euler_angles

        # Velocities and actions
        obs[:, layout.body_linear_velocity] = self.obs_dict["robot_body_linvel"]
        obs[:, layout.body_angular_velocity] = self.obs_dict["robot_body_angvel"]
        obs[:, layout.actions] = self.obs_dict["robot_actions"]

        # VAE latents (drone and static camera)
        if isinstance(self.image_latents, torch.Tensor) and self.image_latents.shape[1] >= 64:
            obs[:, layout.drone_vae_latents] = self.image_latents[:, :64]
        if isinstance(self.static_image_latents, torch.Tensor) and self.static_image_latents.shape[1] >= 64:
            obs[:, layout.static_vae_latents] = self.static_image_latents[:, :64]

        # (Removed W&B latent stats logging per request)

        # Final observation NaN/Inf guard: sanitize outgoing observations tensor
        obs_tensor = self.task_obs.get("observations", None)
        if isinstance(obs_tensor, torch.Tensor):
            bad = torch.isnan(obs_tensor) | torch.isinf(obs_tensor)
            if torch.any(bad):
                if self.task_config.guard_debug_enabled:
                    logger.warning(f"[NaNGuard] Sanitizing {int(torch.sum(bad).item())} invalid obs entries before return.")
                obs_tensor[bad] = 0.0

    def compute_rewards_and_crashes(self, obs_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute rewards with gate-specific components."""
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        
        self.pos_error_vehicle_frame_prev[:] = self.pos_error_vehicle_frame
        self.pos_error_vehicle_frame[:] = quat_rotate_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        
        # obs_dict contains direct references to global tensors that get updated simultaneously
        current_actions = obs_dict["robot_actions"].clone()
        previous_actions = obs_dict["robot_prev_actions"].clone()
        
        # First-step stabilization: align previous error/actions with current for fresh episodes
        prev_actions_for_reward = previous_actions
        fresh_mask = self._episode_fresh
        if isinstance(fresh_mask, torch.Tensor) and fresh_mask.shape[0] == self.num_envs:
            if torch.any(fresh_mask):
                # Set previous error equal to current on the first step after reset
                self.pos_error_vehicle_frame_prev[fresh_mask] = self.pos_error_vehicle_frame[fresh_mask]
                # Also zero action diff on the first step after reset
                prev_actions_for_reward = previous_actions.clone()
                prev_actions_for_reward[fresh_mask] = current_actions[fresh_mask]
        
        # Curriculum multiplier ablation: pass effective fraction to scripted reward
        cm_disabled = read_env_bool("SF_DISABLE_CURRICULUM_MULTIPLIER", self.task_config.disable_curriculum_multiplier)
        if not cm_disabled:
            cm_disabled = bool(self.task_config.disable_curriculum_multiplier)
        try:
            frac_current = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)
        except Exception:
            frac_current = 0.0
        frac_eff = 0.0 if cm_disabled else float(frac_current)
        self._curriculum_multiplier_factor = 1.0 + 0.5 * frac_eff

        boundary_violation_one_shot_mask = self._detect_boundary_violation(robot_position)

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
        rp = self.task_config.reward_parameters
        try:
            H = int(self.task_config.episode_len_steps)
            if H <= 0:
                H = 100
        except (ValueError, TypeError):
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
        except (ValueError, TypeError):
            s = torch.full((self.num_envs,), 1.0 / float(H), device=self.device)
        raw_time_penalty = -lam0 * (1.0 + lam1 * torch.pow(s, p))
        reward_scale = float(rp.get('reward_scale', 0.1))
        try:
            mult_factor = float(self._curriculum_multiplier_factor)
        except (ValueError, TypeError):
            mult_factor = 1.0
        time_penalty = (mult_factor * reward_scale * raw_time_penalty)
        rewards = rewards + time_penalty
        # Accumulate for per-episode debugging
        non_terminated = ~self.terminations
        if torch.sum(non_terminated) > 0:
            self.episode_time_penalty[non_terminated] += time_penalty[non_terminated]

        # Static camera FOV visibility reward (depth-based frustum check, shaped)
        try:
            try:
                fov_mag = float(self.task_config.reward_parameters.get("static_fov_visibility_reward_magnitude", 0.0))
            except (ValueError, TypeError):
                fov_mag = 0.0
            # If env var SF_ENABLE_STATIC_FOV_REWARD is not explicitly true, force-disable
            try:
                _env_flag = os.environ.get('SF_ENABLE_STATIC_FOV_REWARD', '').strip().lower()
                if _env_flag not in ('1', 'true', 'yes', 'y'):
                    fov_mag = 0.0
            except (KeyError, TypeError):
                fov_mag = 0.0
            try:
                fov_alpha = float(self.task_config.reward_parameters.get("static_fov_visibility_exponent", 2.0))
            except (ValueError, TypeError):
                fov_alpha = 2.0
            if fov_mag != 0.0:
                # Camera base position (x=0, y=base_y, z either adaptive gate center or fixed 1.5)
                try:
                    parent = self.sim_env
                    gtd = parent.global_tensor_dict if (parent is not None and hasattr(parent, 'global_tensor_dict')) else {}
                except (AttributeError, KeyError):
                    gtd = {}
                try:
                    base_y = float(os.environ.get('SF_STATIC_CAMERA_BASE_Y', gtd.get('static_camera/base_y', -3.0)))
                except (ValueError, TypeError):
                    base_y = -3.0
                try:
                    base_z_env = os.environ.get('SF_STATIC_CAMERA_BASE_Z', None)
                    if base_z_env is None:
                        base_z_env = gtd.get('static_camera/base_z', 1.5)
                    adaptive_z = isinstance(base_z_env, str) and base_z_env.strip().lower() == 'adaptive'
                except (KeyError, TypeError):
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
                    self.episode_static_fov_visibility_reward[add_mask] += fov_reward[add_mask]

                # Periodic debug infos
                if(self.num_task_steps % 200 == 0):
                    frac_visible = float(torch.mean(visible.float()).item())
                    avg_h = float(torch.mean(horiz_angle).item())
                    avg_v = float(torch.mean(vert_angle).item())
                    avg_score = float(torch.mean(fov_score).item())
                    self.infos["static_fov/visible_fraction"] = torch.tensor(frac_visible, dtype=torch.float32)
                    self.infos["static_fov/avg_horiz_angle_rad"] = torch.tensor(avg_h, dtype=torch.float32)
                    self.infos["static_fov/avg_vert_angle_rad"] = torch.tensor(avg_v, dtype=torch.float32)
                    self.infos["static_fov/avg_score"] = torch.tensor(avg_score, dtype=torch.float32)

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
                                scm = self.static_camera_manager
                                yaw_cur = float(scm.current_camera_angles[env0]) if (scm is not None and hasattr(scm, 'current_camera_angles') and len(scm.current_camera_angles) > env0) else 0.0
                            except (ValueError, TypeError):
                                yaw_cur = 0.0
                            logger.warning(f" 🖼️ env0 FOV: visible={1 if vis0 else 0} h={hdeg0:.1f}° v={vdeg0:.1f}° | score={score0:.3f} | cam_yaw={yaw_cur:.1f}° | x_c={xc0:.2f}, y_c={yc0:.2f}, z_c={zc0:.2f}")
                    except (ValueError, TypeError):
                        pass
        except (ValueError, TypeError):
            pass
        # UPDATE EPISODE REWARD TRACKING: Track cumulative reward components
        self.update_episode_reward_tracking(obs_dict, rewards, crashes)
        self._log_comprehensive_reward_debug(obs_dict, rewards, crashes, boundary_violation_one_shot_mask, camera_gate_alignment)
        
        # Store camera alignment for debugging
        self.camera_alignment_debug = camera_gate_alignment
        
        return rewards, crashes, camera_gate_alignment
    def _detect_boundary_violation(self, robot_position: torch.Tensor) -> torch.Tensor:
        """Detect one-shot boundary violations (crossing gate plane outside passage window)."""
        y_margin = 0.2
        behind_gate_mask = robot_position[:, 1] > (self.gate_position[:, 1] + y_margin)
        gate_passage_width_tolerance = self.gate_width * 0.5
        gate_min_height = self.gate_position[:, 2] + self.gate_height * 0.0
        gate_max_height = self.gate_position[:, 2] + self.gate_height * 1.0
        within_passage_window = (
            (torch.abs(robot_position[:, 0] - self.gate_position[:, 0]) < gate_passage_width_tolerance)
            & (robot_position[:, 2] > gate_min_height)
            & (robot_position[:, 2] < gate_max_height)
        )
        misaligned_cross_mask = behind_gate_mask & (~within_passage_window) & (~self.gate_passed)
        if not True:
            self._bv_flag_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        boundary_violation_one_shot_mask = misaligned_cross_mask & (~self._bv_flag_episode)
        self._bv_flag_episode |= boundary_violation_one_shot_mask
        return boundary_violation_one_shot_mask


        



