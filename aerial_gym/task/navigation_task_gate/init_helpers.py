from __future__ import annotations

import os

import numpy as np
import torch
from gym.spaces import Box, Dict

from aerial_gym.task.navigation_task_gate.init_curriculum_logging import (
    log_initial_curriculum_state,
)
from aerial_gym.task.schemas import EpisodeRewardAccumulators, EpisodeTrajectoryState
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.vae.vae_image_encoder import VAEImageEncoder

logger = CustomLogger("navigation_task_gate_init")


class InitHelpers:
    def __init__(self, task: object) -> None:
        self.task = task

    def _init_gate_tracking_tensors(self) -> None:
        """Initialise gate-specific tracking tensors and adaptive dimensions."""
        self.task.gate_position = torch.zeros(
            (self.task.sim_env.num_envs, 3), device=self.task.device
        )
        self.task.gate_approach_distance = torch.zeros(
            self.task.sim_env.num_envs, device=self.task.device
        )
        self.task.gate_width = torch.zeros((self.task.sim_env.num_envs,), device=self.task.device)
        self.task.gate_height = torch.zeros((self.task.sim_env.num_envs,), device=self.task.device)
        self.task.gate_center_height = torch.zeros(
            (self.task.sim_env.num_envs,), device=self.task.device
        )
        self.task.gate_scale_factors = torch.ones(
            (self.task.sim_env.num_envs,), device=self.task.device
        )

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
        self.task.action_transformation_function = (
            self.task.task_config.action_transformation_function
        )
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
        self.task.episode_rewards = EpisodeRewardAccumulators.create(
            self.task.num_envs, self.task.device
        )
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
        self.task.episode_static_fov_visibility_reward = (
            self.task.episode_rewards.static_fov_visibility_reward
        )
        self.task.episode_boundary_violation_penalty = (
            self.task.episode_rewards.boundary_violation_penalty
        )
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
        forced = os.environ.get("SF_FORCE_CURRICULUM_LEVEL", None)
        if forced is None:
            forced = self.task.task_config.force_curriculum_level
        if forced is not None:
            self.task.curriculum_level = int(forced)
        self.task.obs_dict.curriculum_level = self.task.curriculum_level

        # Track maximum curriculum level reached (for no-decrease policy)
        self.task.max_curriculum_level_reached = self.task.curriculum_level

        # Get obstacle count using the already-set curriculum level
        obstacles_behind_gate = self.task.task_config.curriculum.get_obstacle_count_behind_gate(
            self.task.curriculum_level
        )

        # Initialize curriculum logging first
        self.task.curriculum_log_file = None  # Initialize to None
        self.task._curriculum.setup_curriculum_logging()  # Set up logging before using it

        # FIXED CALCULATION: Account for visible assets only (not all loaded assets)
        # Even though 11 gate variants are loaded, only 1 is visible at any time
        visible_gates = 1  # Only 1 gate visible at a time (others hidden by gate selection system)
        walls = 6  # 6 boundary walls
        robot = 1  # 1 robot
        fixed_assets_visible = visible_gates + walls + robot  # = 8 visible fixed assets

        total_obstacles_in_env = fixed_assets_visible + obstacles_behind_gate

        logger.info(
            f"Curriculum L{self.task.curriculum_level}: "
            f"{fixed_assets_visible} fixed + {obstacles_behind_gate} curriculum = {total_obstacles_in_env} total assets"
        )

        # Update observation dictionary and global tensor dict with obstacle count
        self.task.obs_dict.num_obstacles_in_env = total_obstacles_in_env
        old_count = self.task.sim_env.global_tensor_dict.num_obstacles_in_env
        if old_count != total_obstacles_in_env:
            logger.info(f"Updated obstacle count: {old_count} -> {total_obstacles_in_env}")
            self.task.sim_env.global_tensor_dict.num_obstacles_in_env = total_obstacles_in_env

        # Initialize camera difficulty parameters (only static camera curriculum remains)
        (
            self.task.max_camera_angle,
            self.task.camera_height_offset,
            self.task.camera_distance_offset,
        ) = self.task.task_config.curriculum.get_static_camera_difficulty(
            self.task.curriculum_level
        )

        # Calculate progress fraction
        self.task.curriculum_progress_fraction = (
            self.task.curriculum_level - self.task.task_config.curriculum.min_level
        ) / (
            self.task.task_config.curriculum.max_level - self.task.task_config.curriculum.min_level
        )

        log_initial_curriculum_state(
            self.task, fixed_assets_visible, obstacles_behind_gate, total_obstacles_in_env
        )
