"""Typed replacement for global_tensor_dict.

Supports both dict-style and attribute-style access for backward compatibility.
Dict-style access (obj["key"]) delegates to attributes so existing code keeps working
while new code uses obj.key for autocomplete and type safety.

Usage:
    state = TensorState()
    state["robot_position"] = tensor  # backward compat (dict-style)
    state.robot_position = tensor     # preferred (attribute-style)
    pos = state.robot_position        # typed, IDE autocomplete
    pos = state["robot_position"]     # still works
    val = state.get("key", default)   # still works
"""

from __future__ import annotations

from typing import List, Optional

import torch


class TensorState:
    """Central simulation state container — typed replacement for global_tensor_dict.

    All 69+ attributes are initialized to None. Supports dict-style access
    via __getitem__/__setitem__/get/__contains__ for backward compatibility.
    """

    def __init__(self) -> None:
        # ── Robot state ──────────────────────────────────────────────
        self.robot_state_tensor: Optional[torch.Tensor] = None  # (num_envs, 13)
        self.robot_position: Optional[torch.Tensor] = None  # (num_envs, 3)
        self.robot_orientation: Optional[torch.Tensor] = None  # (num_envs, 4)
        self.robot_linvel: Optional[torch.Tensor] = None  # (num_envs, 3)
        self.robot_angvel: Optional[torch.Tensor] = None  # (num_envs, 3)
        self.robot_body_angvel: Optional[torch.Tensor] = None  # (num_envs, 3)
        self.robot_body_linvel: Optional[torch.Tensor] = None  # (num_envs, 3)
        self.robot_euler_angles: Optional[torch.Tensor] = None  # (num_envs, 3)
        self.robot_vehicle_orientation: Optional[torch.Tensor] = None
        self.robot_vehicle_linvel: Optional[torch.Tensor] = None
        self.robot_mass: Optional[torch.Tensor] = None  # (num_envs,)
        self.robot_inertia: Optional[torch.Tensor] = None  # (num_envs, 3, 3)

        # ── Asset state ──────────────────────────────────────────────
        self.vec_root_tensor: Optional[torch.Tensor] = None  # (num_envs, num_assets, 13)
        self.env_asset_state_tensor: Optional[torch.Tensor] = None
        self.unfolded_env_asset_state_tensor: Optional[torch.Tensor] = None
        self.unfolded_env_asset_state_tensor_const: Optional[torch.Tensor] = None
        self.obstacle_position: Optional[torch.Tensor] = None
        self.obstacle_orientation: Optional[torch.Tensor] = None
        self.obstacle_linvel: Optional[torch.Tensor] = None
        self.obstacle_angvel: Optional[torch.Tensor] = None
        self.obstacle_body_angvel: Optional[torch.Tensor] = None
        self.obstacle_body_linvel: Optional[torch.Tensor] = None
        self.obstacle_euler_angles: Optional[torch.Tensor] = None

        # ── Force and contact ────────────────────────────────────────
        self.rigid_body_state_tensor: Optional[torch.Tensor] = None
        self.global_force_tensor: Optional[torch.Tensor] = None  # (total_rigid_bodies, 3)
        self.global_torque_tensor: Optional[torch.Tensor] = None
        self.robot_force_tensor: Optional[torch.Tensor] = None
        self.robot_torque_tensor: Optional[torch.Tensor] = None
        self.global_contact_force_tensor: Optional[torch.Tensor] = None
        self.robot_contact_force_tensor: Optional[torch.Tensor] = None
        self.obstacle_force_tensor: Optional[torch.Tensor] = None
        self.obstacle_torque_tensor: Optional[torch.Tensor] = None

        # ── DOF state ────────────────────────────────────────────────
        self.dof_control_mode: str = "none"
        self.unfolded_dof_state_tensor: Optional[torch.Tensor] = None
        self.dof_state_tensor: Optional[torch.Tensor] = None
        self.dof_position_setpoint_tensor: Optional[torch.Tensor] = None
        self.dof_velocity_setpoint_tensor: Optional[torch.Tensor] = None
        self.dof_effort_tensor: Optional[torch.Tensor] = None

        # ── Sensor outputs ───────────────────────────────────────────
        self.depth_range_pixels: Optional[torch.Tensor] = None
        self.segmentation_pixels: Optional[torch.Tensor] = None
        self.rgb_pixels: Optional[torch.Tensor] = None
        self.force_sensor_tensor: Optional[torch.Tensor] = None
        self.imu_measurement: Optional[torch.Tensor] = None

        # ── Environment metadata ─────────────────────────────────────
        self.env_bounds_min: Optional[torch.Tensor] = None  # (num_envs, 3)
        self.env_bounds_max: Optional[torch.Tensor] = None  # (num_envs, 3)
        self.gravity: Optional[torch.Tensor] = None  # (num_envs, 3)
        self.dt: float = 0.0
        self.sim_steps: Optional[torch.Tensor] = None  # (num_envs,) int32

        # ── Actions ──────────────────────────────────────────────────
        self.robot_actions: Optional[torch.Tensor] = None
        self.robot_prev_actions: Optional[torch.Tensor] = None
        self.num_robot_actions: int = 0
        self.env_actions: Optional[torch.Tensor] = None
        self.prev_env_actions: Optional[torch.Tensor] = None
        self.num_env_actions: int = 0

        # ── Episode status ───────────────────────────────────────────
        self.crashes: Optional[torch.Tensor] = None  # (num_envs,) bool
        self.terminations: Optional[torch.Tensor] = None  # (num_envs,) bool
        self.truncations: Optional[torch.Tensor] = None  # (num_envs,) bool

        # ── Gate variant tracking ────────────────────────────────────
        self.gate_variant_counter: Optional[torch.Tensor] = None
        self.active_gate_variant_index: Optional[torch.Tensor] = None
        self.active_gate_variant_array_index: Optional[torch.Tensor] = None
        self.gate_variant_indices_per_env: List[List[int]] = []
        self.gate_variant_names_per_env: List[List[str]] = []

        # ── Asset state ratios ───────────────────────────────────────
        self.asset_min_state_ratio: Optional[torch.Tensor] = None
        self.asset_max_state_ratio: Optional[torch.Tensor] = None

        # ── Curriculum and randomization ─────────────────────────────
        self.curriculum_level: int = 0
        self.eval_stretch_enabled: bool = False
        self.eval_stretch_end_level: int = 33
        self.num_obstacles_in_env: int = 0

        # ── Gate geometry ────────────────────────────────────────────
        self.gate_position: Optional[torch.Tensor] = None

        # ── Warp rendering ───────────────────────────────────────────
        self.CONST_WARP_MESH_ID_LIST: Optional[list] = None
        self.CONST_WARP_MESH_PER_ENV: Optional[list] = None
        self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR: Optional[torch.Tensor] = None
        self.VERTEX_MAPS_PER_ENV_ORIGINAL: Optional[torch.Tensor] = None

    # ── Dict-style backward compatibility ────────────────────────────

    def __getitem__(self, key: str) -> object:
        try:
            return self.__dict__[key]
        except KeyError:
            raise KeyError(f"TensorState has no key '{key}'") from None

    def __setitem__(self, key: str, value: object) -> None:
        self.__dict__[key] = value

    def get(self, key: str, default: object = None) -> object:
        return self.__dict__.get(key, default)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self.__dict__ and self.__dict__[key] is not None

    def __iter__(self):
        return iter(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, key: str, *args: object) -> object:
        return self.__dict__.pop(key, *args)

    def update(self, other: dict) -> None:
        self.__dict__.update(other)
