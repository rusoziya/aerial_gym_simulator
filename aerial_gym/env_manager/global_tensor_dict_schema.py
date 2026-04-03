"""TypedDict schema for global_tensor_dict and global_sim_dict.

Provides type-checked dict access — IDEs and type checkers catch key typos
like dict["robot_positon"] at development time instead of runtime.

Usage:
    from aerial_gym.env_manager.global_tensor_dict_schema import GlobalTensorDict
    self.global_tensor_dict: GlobalTensorDict = {}
"""

from __future__ import annotations

from typing import List, Optional

import torch
from typing_extensions import TypedDict


class GlobalTensorDict(TypedDict, total=False):
    """Typed schema for global_tensor_dict — the central simulation state container.

    All keys are optional (total=False) because the dict is populated incrementally
    during environment setup. Keys marked with # REQUIRED are always present after
    env_manager.prepare_sim() completes.
    """

    # Robot state (populated by tensor_population.py + robot_manager.py)
    robot_state_tensor: torch.Tensor  # REQUIRED — (num_envs, 13)
    robot_position: torch.Tensor  # REQUIRED — (num_envs, 3)
    robot_orientation: torch.Tensor  # REQUIRED — (num_envs, 4) quaternion
    robot_linvel: torch.Tensor  # REQUIRED — (num_envs, 3) world frame
    robot_angvel: torch.Tensor  # REQUIRED — (num_envs, 3) world frame
    robot_body_angvel: torch.Tensor  # REQUIRED — (num_envs, 3) body frame
    robot_body_linvel: torch.Tensor  # REQUIRED — (num_envs, 3) body frame
    robot_euler_angles: torch.Tensor  # REQUIRED — (num_envs, 3)
    robot_vehicle_orientation: torch.Tensor  # REQUIRED — shape varies by robot
    robot_vehicle_linvel: torch.Tensor  # REQUIRED — shape varies by robot
    robot_mass: torch.Tensor  # REQUIRED — (num_envs,)
    robot_inertia: torch.Tensor  # REQUIRED — (num_envs, 3, 3)

    # Asset state (populated by IGE_env_manager.py)
    vec_root_tensor: torch.Tensor  # REQUIRED — (num_envs, num_assets, 13)
    env_asset_state_tensor: torch.Tensor  # REQUIRED — (num_envs, num_assets-1, 13)
    unfolded_env_asset_state_tensor: torch.Tensor  # REQUIRED
    unfolded_env_asset_state_tensor_const: torch.Tensor  # REQUIRED — reference copy
    obstacle_position: torch.Tensor  # (num_envs, num_obs, 3) — if obstacles present
    obstacle_orientation: torch.Tensor  # (num_envs, num_obs, 4)
    obstacle_linvel: torch.Tensor
    obstacle_angvel: torch.Tensor
    obstacle_body_angvel: torch.Tensor
    obstacle_body_linvel: torch.Tensor
    obstacle_euler_angles: torch.Tensor

    # Force and contact (populated by tensor_population.py + IGE_env_manager.py)
    rigid_body_state_tensor: torch.Tensor  # REQUIRED
    global_force_tensor: torch.Tensor  # REQUIRED — (total_rigid_bodies, 3)
    global_torque_tensor: torch.Tensor  # REQUIRED — (total_rigid_bodies, 3)
    robot_force_tensor: torch.Tensor  # REQUIRED — (num_envs, num_robot_bodies, 3)
    robot_torque_tensor: torch.Tensor  # REQUIRED
    global_contact_force_tensor: torch.Tensor  # REQUIRED
    robot_contact_force_tensor: torch.Tensor  # REQUIRED — (num_envs, 3)
    obstacle_force_tensor: torch.Tensor  # if obstacles present
    obstacle_torque_tensor: torch.Tensor  # if obstacles present

    # DOF state (populated by IGE_env_manager.py — if robot has DOFs)
    dof_control_mode: str
    unfolded_dof_state_tensor: torch.Tensor
    dof_state_tensor: torch.Tensor
    dof_position_setpoint_tensor: torch.Tensor
    dof_velocity_setpoint_tensor: torch.Tensor
    dof_effort_tensor: torch.Tensor

    # Sensor outputs (populated by robot_manager.py / warp sensors)
    depth_range_pixels: torch.Tensor  # (num_envs, num_cams, H, W)
    segmentation_pixels: torch.Tensor  # (num_envs, num_cams, H, W)
    rgb_pixels: torch.Tensor  # (num_envs, num_cams, H, W, 4)
    force_sensor_tensor: torch.Tensor
    imu_measurement: torch.Tensor

    # Environment metadata (populated by IGE_env_manager.py)
    env_bounds_min: torch.Tensor  # REQUIRED — (num_envs, 3)
    env_bounds_max: torch.Tensor  # REQUIRED — (num_envs, 3)
    gravity: torch.Tensor  # REQUIRED — (num_envs, 3)
    dt: float  # REQUIRED — physics timestep

    # Actions (populated by robot_manager.py + env_manager.py)
    robot_actions: torch.Tensor  # REQUIRED — (num_envs, num_robot_actions)
    robot_prev_actions: torch.Tensor  # REQUIRED
    num_robot_actions: int  # REQUIRED
    env_actions: Optional[torch.Tensor]  # lazy-initialized
    prev_env_actions: Optional[torch.Tensor]
    num_env_actions: int  # REQUIRED

    # Episode status (populated by env_manager.py)
    crashes: torch.Tensor  # REQUIRED — (num_envs,) bool
    terminations: torch.Tensor  # REQUIRED — (num_envs,) bool
    truncations: torch.Tensor  # REQUIRED — (num_envs,) bool
    sim_steps: torch.Tensor  # (num_envs,) int32

    # Gate variant tracking (populated by env_manager.py)
    gate_variant_counter: torch.Tensor  # (num_envs,) int64
    active_gate_variant_index: torch.Tensor  # (num_envs,) long
    active_gate_variant_array_index: torch.Tensor  # (num_envs,) long
    gate_variant_indices_per_env: List[List[int]]
    gate_variant_names_per_env: List[List[str]]

    # Asset state ratios (populated by env_manager.py)
    asset_min_state_ratio: torch.Tensor  # (num_envs, assets_per_env, 13)
    asset_max_state_ratio: torch.Tensor

    # Curriculum and randomization flags (set by task / env vars)
    curriculum_level: int
    eval_stretch_enabled: bool
    eval_stretch_end_level: int
    num_obstacles_in_env: int

    # Gate geometry (set by navigation task)
    gate_position: torch.Tensor

    # Warp rendering (set by warp_env_manager.py)
    CONST_WARP_MESH_ID_LIST: Optional[list]
    CONST_WARP_MESH_PER_ENV: Optional[list]
    CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR: Optional[torch.Tensor]
    VERTEX_MAPS_PER_ENV_ORIGINAL: Optional[torch.Tensor]

    # Static camera state (set by static_camera_manager / env vars)


class GlobalSimDict(TypedDict, total=False):
    """Typed schema for global_sim_dict — Isaac Gym simulation references."""

    gym: object  # gymapi.Gym
    sim: object  # gymapi.Sim
    env_cfg: type  # environment config class
    sim_cfg: type  # simulation config class
    use_warp: bool
    num_envs: int
    robot_config: type  # robot config class
