"""Typed dataclass schemas for navigation task data structures.

These schemas replace raw dicts and magic index slices with named,
typed, self-documenting structures. They serve as both runtime
containers and type annotations throughout the task code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Observation Layout — describes the 150D gate navigation observation vector


@dataclass(frozen=True)
class ObservationLayout:
    """Index ranges for the 150D gate navigation observation vector.

    Usage:
        layout = ObservationLayout()
        obs[:, layout.drone_position] = drone_pos
        latents = obs[:, layout.drone_vae_latents]
    """

    drone_position: slice = field(default_factory=lambda: slice(0, 3))  # 3D
    static_camera_position: slice = field(default_factory=lambda: slice(3, 6))  # 3D
    static_camera_orientation: slice = field(default_factory=lambda: slice(6, 9))  # 3D
    drone_orientation: slice = field(default_factory=lambda: slice(9, 12))  # 3D
    body_linear_velocity: slice = field(default_factory=lambda: slice(12, 15))  # 3D
    body_angular_velocity: slice = field(default_factory=lambda: slice(15, 18))  # 3D
    actions: slice = field(default_factory=lambda: slice(18, 22))  # 4D
    drone_vae_latents: slice = field(default_factory=lambda: slice(22, 86))  # 64D
    static_vae_latents: slice = field(default_factory=lambda: slice(86, 150))  # 64D

    total_dim: int = 150


# Singleton for use throughout the codebase
GATE_OBS_LAYOUT = ObservationLayout()


# Episode Reward Tracking — replaces 12+ separate self.episode_* tensors


@dataclass
class EpisodeRewardAccumulators:
    """Per-environment accumulators for reward components within an episode.

    All tensors have shape (num_envs,) and live on the task's device.
    Reset to zero at episode boundaries via reset_envs().
    """

    pos_reward: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    very_close_reward: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    getting_closer_reward: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    gate_approach_reward: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    gate_alignment_reward: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    camera_facing_reward: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    action_penalty: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    gate_passage_reward: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    collision_penalty: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    image_reward: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    static_fov_visibility_reward: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    boundary_violation_penalty: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    time_penalty: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    timeout_penalty: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    lengths: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    @classmethod
    def create(cls, num_envs: int, device: str) -> EpisodeRewardAccumulators:
        """Create zero-initialized accumulators for the given environment count."""
        return cls(
            pos_reward=torch.zeros(num_envs, device=device),
            very_close_reward=torch.zeros(num_envs, device=device),
            getting_closer_reward=torch.zeros(num_envs, device=device),
            gate_approach_reward=torch.zeros(num_envs, device=device),
            gate_alignment_reward=torch.zeros(num_envs, device=device),
            camera_facing_reward=torch.zeros(num_envs, device=device),
            action_penalty=torch.zeros(num_envs, device=device),
            gate_passage_reward=torch.zeros(num_envs, device=device),
            collision_penalty=torch.zeros(num_envs, device=device),
            image_reward=torch.zeros(num_envs, device=device),
            static_fov_visibility_reward=torch.zeros(num_envs, device=device),
            boundary_violation_penalty=torch.zeros(num_envs, device=device),
            time_penalty=torch.zeros(num_envs, device=device),
            timeout_penalty=torch.zeros(num_envs, device=device),
            lengths=torch.zeros(num_envs, device=device),
        )

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset accumulators for the given environment indices."""
        self.pos_reward[env_ids] = 0
        self.very_close_reward[env_ids] = 0
        self.getting_closer_reward[env_ids] = 0
        self.gate_approach_reward[env_ids] = 0
        self.gate_alignment_reward[env_ids] = 0
        self.camera_facing_reward[env_ids] = 0
        self.action_penalty[env_ids] = 0
        self.gate_passage_reward[env_ids] = 0
        self.collision_penalty[env_ids] = 0
        self.image_reward[env_ids] = 0
        self.static_fov_visibility_reward[env_ids] = 0
        self.boundary_violation_penalty[env_ids] = 0
        self.time_penalty[env_ids] = 0
        self.timeout_penalty[env_ids] = 0
        self.lengths[env_ids] = 0


# Episode Trajectory State — replaces scattered _ep_* tensors


@dataclass
class EpisodeTrajectoryState:
    """Per-environment trajectory tracking within an episode.

    All tensors have shape (num_envs,) or (num_envs, 3).
    Reset at episode boundaries.
    """

    fresh: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    spawn_pos: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    gate_center_at_spawn: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    last_pos: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    path_len: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    steps: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    min_gate_dist: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    gate_crossed: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    time_to_gate: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    center_offset_cross: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    height_offset_cross: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    bv_flag_episode: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    @classmethod
    def create(cls, num_envs: int, device: str) -> EpisodeTrajectoryState:
        """Create initialized trajectory state for the given environment count."""
        return cls(
            fresh=torch.ones(num_envs, dtype=torch.bool, device=device),
            spawn_pos=torch.zeros((num_envs, 3), device=device),
            gate_center_at_spawn=torch.zeros((num_envs, 3), device=device),
            last_pos=torch.zeros((num_envs, 3), device=device),
            path_len=torch.zeros(num_envs, device=device),
            steps=torch.zeros(num_envs, dtype=torch.long, device=device),
            min_gate_dist=torch.full((num_envs,), float("inf"), device=device),
            gate_crossed=torch.zeros(num_envs, dtype=torch.bool, device=device),
            time_to_gate=torch.full((num_envs,), float("nan"), device=device),
            center_offset_cross=torch.full((num_envs,), float("nan"), device=device),
            height_offset_cross=torch.full((num_envs,), float("nan"), device=device),
            bv_flag_episode=torch.zeros(num_envs, dtype=torch.bool, device=device),
        )

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset trajectory state for the given environment indices."""
        self.fresh[env_ids] = True
        self.spawn_pos[env_ids] = 0
        self.gate_center_at_spawn[env_ids] = 0
        self.last_pos[env_ids] = 0
        self.path_len[env_ids] = 0
        self.steps[env_ids] = 0
        self.min_gate_dist[env_ids] = float("inf")
        self.gate_crossed[env_ids] = False
        self.time_to_gate[env_ids] = float("nan")
        self.center_offset_cross[env_ids] = float("nan")
        self.height_offset_cross[env_ids] = float("nan")
        self.bv_flag_episode[env_ids] = False
