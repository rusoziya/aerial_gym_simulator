from __future__ import annotations

import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY
from aerial_gym.config.asset_config.base_asset import BaseAssetParams

GATE_SEMANTIC_ID: int = 10


class GateAssetConfig:
    class gate_asset_params(BaseAssetParams):
        """Configuration for gate assets that drones can fly through."""

        num_assets: int = 31

        asset_folder: str = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/gates"
        file: str | None = None

        min_position_ratio: list[float] = [0.5, 0.5, 0.0]
        max_position_ratio: list[float] = [0.5, 0.5, 0.0]

        min_euler_angles: list[float] = [0.0, 0.0, np.pi / 2]
        max_euler_angles: list[float] = [0.0, 0.0, np.pi / 2]

        min_state_ratio: list[float] = [
            0.5,
            0.5,
            0.0,
            0.0,
            0.0,
            np.pi / 2,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        max_state_ratio: list[float] = [
            0.5,
            0.5,
            0.0,
            0.0,
            0.0,
            np.pi / 2,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        collision_mask: int = 0
        disable_gravity: bool = True
        fix_base_link: bool = True

        replace_cylinder_with_capsule: bool = True
        collapse_fixed_joints: bool = True

        density: float = 2000.0
        angular_damping: float = 1.0
        linear_damping: float = 1.0
        max_angular_velocity: float = 0.0
        max_linear_velocity: float = 0.0

        color: list[int] | None = [150, 150, 150]

        body_semantic_label: int = GATE_SEMANTIC_ID
        link_semantic_label: int = GATE_SEMANTIC_ID
        per_link_semantic: bool = True
        semantic_id: int = GATE_SEMANTIC_ID

        keep_in_env: bool = False

        place_force_sensor: bool = True
        force_sensor_parent_link: str = "base_link"

        use_collision_mesh_instead_of_visual: bool = False

    class small_gate_asset_params(gate_asset_params):
        """Evaluation-only smaller gates (50-58%)."""

        num_assets: int = 5
        asset_folder: str = (
            f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/smaller gates"
        )
        file: str | None = None


gate_asset_params = GateAssetConfig.gate_asset_params
small_gate_asset_params = GateAssetConfig.small_gate_asset_params
