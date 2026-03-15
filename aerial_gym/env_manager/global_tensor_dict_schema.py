"""Typed schema for global_tensor_dict — the central state container.

Defines the structure of every key in global_tensor_dict as a dataclass.
Used as documentation, type reference, and eventual migration target.

Usage as type annotation:
    self.global_tensor_dict: GlobalTensorDict = {}  # type: ignore[assignment]

Future migration: replace dict["key"] access with dataclass attribute access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class RobotStateTensors:
    """Robot kinematic and dynamic state tensors."""

    robot_state_tensor: torch.Tensor  # (num_envs, 13)
    robot_position: torch.Tensor  # (num_envs, 3) — view into state_tensor
    robot_orientation: torch.Tensor  # (num_envs, 4) — quaternion
    robot_linvel: torch.Tensor  # (num_envs, 3) — world frame
    robot_angvel: torch.Tensor  # (num_envs, 3) — world frame
    robot_body_angvel: torch.Tensor  # (num_envs, 3) — body frame
    robot_body_linvel: torch.Tensor  # (num_envs, 3) — body frame
    robot_euler_angles: torch.Tensor  # (num_envs, 3) — roll, pitch, yaw
    robot_vehicle_orientation: torch.Tensor  # shape varies by robot type
    robot_vehicle_linvel: torch.Tensor  # shape varies by robot type
    robot_mass: torch.Tensor  # (num_envs,)
    robot_inertia: torch.Tensor  # (num_envs, 3, 3)


@dataclass
class AssetStateTensors:
    """Environment asset (obstacle, wall, gate) state tensors."""

    vec_root_tensor: torch.Tensor  # (num_envs, num_assets, 13)
    env_asset_state_tensor: torch.Tensor  # (num_envs, num_assets-1, 13)
    unfolded_env_asset_state_tensor: torch.Tensor  # (num_envs * num_assets, 13)
    unfolded_env_asset_state_tensor_const: torch.Tensor  # clone of above (reference)
    obstacle_position: Optional[torch.Tensor] = None  # (num_envs, num_obs, 3)
    obstacle_orientation: Optional[torch.Tensor] = None  # (num_envs, num_obs, 4)
    obstacle_linvel: Optional[torch.Tensor] = None  # (num_envs, num_obs, 3)
    obstacle_angvel: Optional[torch.Tensor] = None  # (num_envs, num_obs, 3)
    obstacle_body_angvel: Optional[torch.Tensor] = None  # (num_envs, num_obs, 3)
    obstacle_body_linvel: Optional[torch.Tensor] = None  # (num_envs, num_obs, 3)
    obstacle_euler_angles: Optional[torch.Tensor] = None  # (num_envs, num_obs, 3)


@dataclass
class ForceTensors:
    """Force, torque, and contact tensors."""

    rigid_body_state_tensor: torch.Tensor  # (total_rigid_bodies, 13)
    global_force_tensor: torch.Tensor  # (total_rigid_bodies, 3)
    global_torque_tensor: torch.Tensor  # (total_rigid_bodies, 3)
    robot_force_tensor: torch.Tensor  # (num_envs, num_robot_bodies, 3)
    robot_torque_tensor: torch.Tensor  # (num_envs, num_robot_bodies, 3)
    global_contact_force_tensor: torch.Tensor  # (num_envs, num_rigid_bodies, 3)
    robot_contact_force_tensor: torch.Tensor  # (num_envs, 3)
    obstacle_force_tensor: Optional[torch.Tensor] = None  # (num_envs, num_obs, 3)
    obstacle_torque_tensor: Optional[torch.Tensor] = None  # (num_envs, num_obs, 3)


@dataclass
class DOFTensors:
    """Degree of freedom state and setpoint tensors."""

    dof_control_mode: str = "none"
    unfolded_dof_state_tensor: Optional[torch.Tensor] = None  # (total_dofs, 2)
    dof_state_tensor: Optional[torch.Tensor] = None  # (num_envs, dofs_per_env, 2)
    dof_position_setpoint_tensor: Optional[torch.Tensor] = None
    dof_velocity_setpoint_tensor: Optional[torch.Tensor] = None
    dof_effort_tensor: Optional[torch.Tensor] = None


@dataclass
class SensorTensors:
    """Camera, LiDAR, and IMU sensor output tensors."""

    depth_range_pixels: Optional[torch.Tensor] = None  # (num_envs, num_cams, H, W)
    segmentation_pixels: Optional[torch.Tensor] = None  # (num_envs, num_cams, H, W)
    rgb_pixels: Optional[torch.Tensor] = None  # (num_envs, num_cams, H, W, 4)
    force_sensor_tensor: Optional[torch.Tensor] = None
    imu_measurement: Optional[torch.Tensor] = None  # (num_envs, 6)


@dataclass
class EnvironmentMetadata:
    """Environment bounds, physics constants, and step counting."""

    env_bounds_min: torch.Tensor  # (num_envs, 3)
    env_bounds_max: torch.Tensor  # (num_envs, 3)
    gravity: torch.Tensor  # (num_envs, 3)
    dt: float  # physics timestep
    sim_steps: Optional[torch.Tensor] = None  # (num_envs,) int32


@dataclass
class ActionTensors:
    """Robot and environment action tensors."""

    robot_actions: torch.Tensor  # (num_envs, num_robot_actions)
    robot_prev_actions: torch.Tensor  # (num_envs, num_robot_actions)
    num_robot_actions: int = 0
    env_actions: Optional[torch.Tensor] = None  # (num_envs, num_env_actions)
    prev_env_actions: Optional[torch.Tensor] = None
    num_env_actions: int = 0


@dataclass
class EpisodeStatus:
    """Per-environment episode termination flags."""

    crashes: torch.Tensor  # (num_envs,) bool
    terminations: torch.Tensor  # (num_envs,) bool
    truncations: torch.Tensor  # (num_envs,) bool


@dataclass
class GateVariantTracking:
    """Gate variant selection and tracking per environment."""

    gate_variant_counter: torch.Tensor  # (num_envs,) int64
    active_gate_variant_index: torch.Tensor  # (num_envs,) long
    active_gate_variant_array_index: torch.Tensor  # (num_envs,) long
    gate_variant_indices_per_env: List[List[int]] = field(default_factory=list)
    gate_variant_names_per_env: List[List[str]] = field(default_factory=list)
    gate_center_height_per_env: Optional[torch.Tensor] = None  # (num_envs,)


@dataclass
class AssetStateRatios:
    """Min/max state ratios for asset randomization."""

    asset_min_state_ratio: torch.Tensor  # (num_envs, num_assets_per_env, 13)
    asset_max_state_ratio: torch.Tensor  # (num_envs, num_assets_per_env, 13)


@dataclass
class RandomizationFlags:
    """Curriculum and randomization control flags (set via env vars, read via .get())."""

    curriculum_level: int = 0
    eval_stretch_enabled: bool = False
    eval_stretch_end_level: int = 33
    num_obstacles_in_env: int = 0


@dataclass
class GlobalSimDict:
    """Typed schema for global_sim_dict — Isaac Gym simulation references."""

    gym: object  # gymapi.Gym (Isaac Gym handle)
    sim: object  # gymapi.Sim (simulation handle)
    env_cfg: object  # environment configuration class
    sim_cfg: object  # simulation configuration class
    use_warp: bool = False
    num_envs: int = 0
    robot_config: Optional[object] = None  # robot configuration class
