from __future__ import annotations

import os
import torch
import numpy as np

from gym.spaces import Dict, Box

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.vae.vae_image_encoder import VAEImageEncoder
from aerial_gym.utils.env_flag_utils import read_env_bool
from aerial_gym.task.schemas import EpisodeRewardAccumulators, EpisodeTrajectoryState

logger = CustomLogger("navigation_task_gate_init")


class InitHelpers:
    def __init__(self, task: object) -> None:
        self.task = task

    def _init_gate_tracking_tensors(self) -> None:
        """Initialise gate-specific tracking tensors and adaptive dimensions."""
        self.task.gate_position = torch.zeros((self.task.sim_env.num_envs, 3), device=self.task.device)
        self.task.gate_approach_distance = torch.zeros(self.task.sim_env.num_envs, device=self.task.device)
        self.task.gate_width = torch.zeros((self.task.sim_env.num_envs,), device=self.task.device)
        self.task.gate_height = torch.zeros((self.task.sim_env.num_envs,), device=self.task.device)
        self.task.gate_center_height = torch.zeros((self.task.sim_env.num_envs,), device=self.task.device)
        self.task.gate_scale_factors = torch.ones((self.task.sim_env.num_envs,), device=self.task.device)

    def _init_vae_model(self) -> None:
        """Set up shared VAE encoder (or identity fallback) and latent buffers."""
        if self.task.task_config.vae_config.use_vae:
            self.task.shared_vae_model = VAEImageEncoder(
                config=self.task.task_config.vae_config, device=self.task.device
            )
            latent_dims = self.task.task_config.vae_config.latent_dims
            self.task.image_latents = torch.zeros(
                (self.task.sim_env.num_envs, latent_dims),
                device=self.task.device,
                requires_grad=False,
            )
            self.task.static_image_latents = torch.zeros(
                (self.task.sim_env.num_envs, latent_dims),
                device=self.task.device,
                requires_grad=False,
            )
        else:
            self.task.shared_vae_model = lambda x: x
            self.task.image_latents = torch.zeros(
                (self.task.sim_env.num_envs, 1), device=self.task.device, requires_grad=False
            )
            self.task.static_image_latents = torch.zeros(
                (self.task.sim_env.num_envs, 1), device=self.task.device, requires_grad=False
            )

    def _init_observation_action_spaces(self) -> None:
        """Define Gymnasium observation and action spaces."""
        self.task.observation_space = Dict(
            {
                "observations": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.task.task_config.observation_space_dim,),
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
        self.task.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.task.action_transformation_function = self.task.task_config.action_transformation_function
        self.task.num_envs = self.task.sim_env.num_envs

    def _init_task_observations(self) -> None:
        """Allocate task observation tensors."""
        self.task.task_obs = {
            "observations": torch.zeros(
                (self.task.sim_env.num_envs, self.task.task_config.observation_space_dim),
                device=self.task.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.task.sim_env.num_envs,
                    self.task.task_config.privileged_observation_space_dim,
                ),
                device=self.task.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.task.sim_env.num_envs, 1), device=self.task.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.task.sim_env.num_envs, 1), device=self.task.device, requires_grad=False
            ),
        }

    def _init_episode_reward_tracking(self) -> None:
        """Allocate per-component episode reward accumulators and statistics tensors."""
        self.task.episode_rewards = EpisodeRewardAccumulators.create(self.task.num_envs, self.task.device)
        # Backward-compatible aliases (used by update_episode_reward_tracking, reset, logging)
        self.task.episode_pos_reward = self.task.episode_rewards.pos_reward
        self.task.episode_very_close_reward = self.task.episode_rewards.very_close_reward
        self.task.episode_getting_closer_reward = self.task.episode_rewards.getting_closer_reward
        self.task.episode_gate_approach_reward = self.task.episode_rewards.gate_approach_reward
        self.task.episode_gate_alignment_reward = self.task.episode_rewards.gate_alignment_reward
        self.task.episode_camera_facing_reward = self.task.episode_rewards.camera_facing_reward
        self.task.episode_action_penalty = self.task.episode_rewards.action_penalty
        self.task.episode_gate_passage_reward = self.task.episode_rewards.gate_passage_reward
        self.task.episode_collision_penalty = self.task.episode_rewards.collision_penalty
        self.task.episode_image_reward = self.task.episode_rewards.image_reward
        self.task.episode_static_fov_visibility_reward = self.task.episode_rewards.static_fov_visibility_reward
        self.task.episode_boundary_violation_penalty = self.task.episode_rewards.boundary_violation_penalty
        self.task.episode_lengths = self.task.episode_rewards.lengths
        self.task.episode_time_penalty = self.task.episode_rewards.time_penalty
        self.task.episode_timeout_penalty = self.task.episode_rewards.timeout_penalty
        self.task.completed_episodes: list[dict[str, float]] = []
        self.task.max_stored_episodes = 10

    def _init_episode_trajectory_state(self) -> None:
        """Allocate per-environment episode trajectory tracking state."""
        self.task.trajectory = EpisodeTrajectoryState.create(self.task.num_envs, self.task.device)
        # Backward-compatible aliases (used throughout step/reset/curriculum code)
        self.task._episode_fresh = self.task.trajectory.fresh
        self.task._ep_spawn_pos = self.task.trajectory.spawn_pos
        self.task._ep_gate_center_at_spawn = self.task.trajectory.gate_center_at_spawn
        self.task._ep_last_pos = self.task.trajectory.last_pos
        self.task._ep_path_len = self.task.trajectory.path_len
        self.task._ep_steps = self.task.trajectory.steps
        self.task._ep_min_gate_dist = self.task.trajectory.min_gate_dist
        self.task._ep_gate_crossed = self.task.trajectory.gate_crossed
        self.task._ep_time_to_gate = self.task.trajectory.time_to_gate
        self.task._ep_center_offset_cross = self.task.trajectory.center_offset_cross
        self.task._ep_height_offset_cross = self.task.trajectory.height_offset_cross
        self.task._bv_flag_episode = self.task.trajectory.bv_flag_episode
        self.task._bv_prev_env0 = False

    def _init_debug_flags(self) -> None:
        """Initialize one-shot debug/logging flags so hasattr checks are never needed."""
        self.task._drone_cam_debug_last = False
        self.task._drone_vae_debug_last = False
        self.task._static_cam_debug_last = False
        self.task._static_cam_depth_logged = False
        self.task._static_debug_logged = False
        self.task._vae_output_logged = False
        self.task._vae_debug_logged = False
        self.task._no_static_logged = False
        self.task._return_tuple_debug_printed = False
        self.task._final_verification_printed = False
        self.task._prev_drone_depth = None
        self.task._prev_static_depth = None
        self.task._success_window_history = []
        self.task._curriculum_cooldown = 0
        self.task.min_pixel_dist = None
        self.task.curriculum_log_file = None
        self.task._curriculum_multiplier_factor = 1.0
        self.task._debug_cam_world = None
        self.task._debug_drone_eul = None
        self.task._debug_cam_eul = None
        self.task.max_curriculum_level_reached = self.task.curriculum_level
        self.task._last_traj_metrics_per_env = {}
        self.task._last_traj_metrics_avg = {}

    def _init_curriculum(self) -> None:
        """Initialize curriculum level, obstacle counts, camera difficulty, and logging."""
        # Use the curriculum level that was already set during pre-initialization
        # Force curriculum level if requested via env/task config
        forced = os.environ.get('SF_FORCE_CURRICULUM_LEVEL', None)
        if forced is None:
            forced = self.task.task_config.force_curriculum_level
        if forced is not None:
            self.task.curriculum_level = int(forced)
        self.task.obs_dict["curriculum_level"] = self.task.curriculum_level

        # Track maximum curriculum level reached (for no-decrease policy)
        self.task.max_curriculum_level_reached = self.task.curriculum_level

        # Get obstacle count using the already-set curriculum level
        obstacles_behind_gate = self.task.task_config.curriculum.get_obstacle_count_behind_gate(self.task.curriculum_level)

        # Initialize curriculum logging first
        self.task.curriculum_log_file = None  # Initialize to None
        self.task.setup_curriculum_logging()  # Set up logging before using it

        # FIXED CALCULATION: Account for visible assets only (not all loaded assets)
        # Even though 11 gate variants are loaded, only 1 is visible at any time
        visible_gates = 1  # Only 1 gate visible at a time (others hidden by gate selection system)
        walls = 6  # 6 boundary walls  
        robot = 1  # 1 robot
        fixed_assets_visible = visible_gates + walls + robot  # = 8 visible fixed assets

        total_obstacles_in_env = fixed_assets_visible + obstacles_behind_gate

        logger.info(f"CURRICULUM: Visible assets: {visible_gates} gate + {walls} walls + {robot} robot + {obstacles_behind_gate} curriculum = {total_obstacles_in_env} total")
        logger.warning(f"[OBSTACLE_FIX] CURRICULUM: Level {self.task.curriculum_level} should spawn {obstacles_behind_gate} curriculum obstacles")

        # Update observation dictionary with obstacle count
        self.task.obs_dict["num_obstacles_in_env"] = total_obstacles_in_env
        logger.warning(f"[OBSTACLE_DEBUG] Task setting obs_dict num_obstacles_in_env = {total_obstacles_in_env}")

        # Confirm the environment manager has the correct count
        if hasattr(self.task.sim_env, 'global_tensor_dict'):
            old_count = self.task.sim_env.global_tensor_dict.get("num_obstacles_in_env", 0)
            if old_count != total_obstacles_in_env:
                logger.warning(f"[OBSTACLE_DEBUG] MISMATCH: Updating global_tensor_dict from {old_count} to {total_obstacles_in_env}")
                self.task.sim_env.global_tensor_dict["num_obstacles_in_env"] = total_obstacles_in_env
            else:
                logger.warning(f"[OBSTACLE_DEBUG] CONFIRMED: Global tensor dict already has correct obstacle count: {total_obstacles_in_env}")

        logger.info(f"FINAL: Visible assets: {fixed_assets_visible}, Curriculum obstacles: {obstacles_behind_gate}, Total: {total_obstacles_in_env}")

        # Initialize camera difficulty parameters (only static camera curriculum remains)
        self.task.max_camera_angle, self.task.camera_height_offset, self.task.camera_distance_offset = self.task.task_config.curriculum.get_static_camera_difficulty(self.task.curriculum_level)

        logger.info(f"INITIAL CURRICULUM (Level {self.task.curriculum_level}):")
        logger.info(f"   1. OBSTACLES: {obstacles_behind_gate} behind gate (total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)")
        try:
            # Determine baseline level and ablation flags
            baseline_level = int(self.task.task_config.curriculum.min_level)
            pos_dis = False; yaw_dis = False
            gtd = self.task.sim_env.global_tensor_dict
            pos_dis = bool(gtd.get('spawn_randomization/position_disabled', False))
            yaw_dis = bool(gtd.get('spawn_randomization/orientation_disabled', False))
            # Read spawn ranges for active and baseline
            sr_active = self.task.task_config.curriculum.get_spawn_ranges(self.task.curriculum_level)
            sr_base = self.task.task_config.curriculum.get_spawn_ranges(baseline_level)
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
            cam_orient_disabled = bool(self.task.sim_env.global_tensor_dict.get('static_camera_randomization/orientation_disabled', False))
        except (KeyError, TypeError):
            cam_orient_disabled = False
        try:
            dyn_cfg = self.task.task_config.curriculum.enable_dynamic_camera_following
            dyn_dis = bool(self.task.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False))
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
            logger.info(f"   4. CAMERA ANGLE: ±{self.task.max_camera_angle:.1f}deg max range (randomized per episode reset, fixed during episode)")

        # 5. CAMERA NOISE PROGRESSION (D455 Simulation)
        initial_camera_gaussian_std, initial_camera_dropout_rate = self.task.task_config.curriculum.get_camera_noise(self.task.curriculum_level)
        logger.info(f"   5. CAMERA NOISE: Gaussian STD={initial_camera_gaussian_std:.4f}, Dropout={initial_camera_dropout_rate*100:.1f}% (both drone & static)")

        # 6. CAMERA FRAME DROPOUT (entire-frame)
        fd = self.task.task_config.curriculum.get_camera_frame_dropout(self.task.curriculum_level)
        logger.info(f"   6. CAMERA FRAME DROPOUT: drone_total={fd['drone_total']*100:.1f}% (freeze {fd['drone_freeze']*100:.1f}%, blank {fd['drone_blank']*100:.1f}%), static_total={fd['static_total']*100:.1f}% (freeze {fd['static_freeze']*100:.1f}%, blank {fd['static_blank']*100:.1f}%)")

        # 7. STATE NOISE (pose) — new
        state_noise_disabled = False
        try:
            state_noise_disabled = bool(self.task.sim_env.global_tensor_dict.get('state_randomization/noise_disabled', False))
        except (KeyError, TypeError):
            state_noise_disabled = bool(self.task.disable_state_noise_randomization)
        if self.task.task_config.curriculum.enable_state_noise and not state_noise_disabled:
            sn = self.task.task_config.curriculum.get_state_noise(self.task.curriculum_level)
            logger.info(
                f"   7. STATE NOISE: drone_pos_std={sn['drone_pos_std_m']:.4f} m, drone_orient_std={sn['drone_orient_std_rad']*57.2958:.3f} deg, "
                f"static_pos_std={sn['static_pos_std_m']:.4f} m, static_orient_std={sn['static_orient_std_rad']*57.2958:.3f} deg"
            )
        else:
            logger.info("   7. STATE NOISE: disabled")

        logger.info(f"   8. ASSET MANAGER: Updated both obs_dict and global_tensor_dict with count {total_obstacles_in_env}")
        # Curriculum multiplier debug (initial) - compute fraction directly (attribute may not exist yet)
        cm_disabled = read_env_bool("SF_DISABLE_CURRICULUM_MULTIPLIER", self.task.task_config.disable_curriculum_multiplier)
        if not cm_disabled:
            cm_disabled = bool(self.task.task_config.disable_curriculum_multiplier)
        try:
            frac_current = (
                self.task.curriculum_level - self.task.task_config.curriculum.min_level
            ) / (self.task.task_config.curriculum.max_level - self.task.task_config.curriculum.min_level)
        except Exception:
            frac_current = 0.0
        frac_eff = 0.0 if cm_disabled else float(frac_current)
        factor = 1.0 + 0.5 * frac_eff
        logger.info(f"   9. CURRICULUM MULTIPLIER: {'DISABLED' if cm_disabled else 'ENABLED'} (factor={factor:.3f})")

        # Calculate progress fraction
        self.task.curriculum_progress_fraction = (
            self.task.curriculum_level - self.task.task_config.curriculum.min_level
        ) / (self.task.task_config.curriculum.max_level - self.task.task_config.curriculum.min_level)

        logger.info(f"   8. PROGRESS: {self.task.curriculum_progress_fraction:.3f} (level {self.task.curriculum_level}/{self.task.task_config.curriculum.max_level})")
        logger.info(f"   9. EVALUATION: Check every {self.task.task_config.curriculum.check_after_log_instances} instances (success rate threshold: {self.task.task_config.curriculum.success_rate_for_increase:.3f})")

        self.task.log_curriculum_update(f"[INIT] Multi-aspect curriculum initialized at level {self.task.curriculum_level}")


