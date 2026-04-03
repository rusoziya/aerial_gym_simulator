from __future__ import annotations

import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY

THIN_SEMANTIC_ID: int = 1
TREE_SEMANTIC_ID: int = 2
OBJECT_SEMANTIC_ID: int = 3
PANEL_SEMANTIC_ID: int = 20


class asset_state_params:
    num_assets: int = 1

    asset_folder: str = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets"
    file: str | None = None

    min_position_ratio: list[float] = [0.5, 0.5, 0.5]
    max_position_ratio: list[float] = [0.5, 0.5, 0.5]

    collision_mask: int = 1

    disable_gravity: bool = False
    replace_cylinder_with_capsule: bool = True
    flip_visual_attachments: bool = True
    density: float = 0.001
    angular_damping: float = 0.1
    linear_damping: float = 0.1
    max_angular_velocity: float = 100.0
    max_linear_velocity: float = 100.0
    armature: float = 0.001

    collapse_fixed_joints: bool = True
    fix_base_link: bool = True
    specific_filepath: str | None = None
    color: list[int] | None = None
    keep_in_env: bool = False

    body_semantic_label: int = 0
    link_semantic_label: int = 0
    per_link_semantic: bool = False
    semantic_masked_links: dict[str, int] = {}
    place_force_sensor: bool = False
    force_sensor_parent_link: str = "base_link"
    force_sensor_transform: list[float] = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]

    use_collision_mesh_instead_of_visual: bool = False


class panel_asset_params(asset_state_params):
    num_assets: int = 3

    asset_folder: str = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/panels"

    collision_mask: int = 1

    min_position_ratio: list[float] = [0.3, 0.05, 0.05]
    max_position_ratio: list[float] = [0.85, 0.95, 0.95]

    specified_position: list[float] = [
        -1000.0,
        -1000.0,
        -1000.0,
    ]

    min_euler_angles: list[float] = [0.0, 0.0, -np.pi / 3.0]
    max_euler_angles: list[float] = [0.0, 0.0, np.pi / 3.0]

    min_state_ratio: list[float] = [
        0.3,
        0.05,
        0.05,
        0.0,
        0.0,
        -np.pi / 3.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio: list[float] = [
        0.85,
        0.95,
        0.95,
        0.0,
        0.0,
        np.pi / 3.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env: bool = True

    collapse_fixed_joints: bool = True
    per_link_semantic: bool = False
    semantic_id: int = -1
    color: list[int] | None = [170, 66, 66]


class tile_asset_params(asset_state_params):
    num_assets: int = 1

    asset_folder: str = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/tile_meshes"

    collision_mask: int = 1

    min_position_ratio: list[float] = [0.3, 0.05, 0.05]
    max_position_ratio: list[float] = [0.85, 0.95, 0.95]

    specified_position: list[float] = [
        -1000.0,
        -1000.0,
        -1000.0,
    ]

    min_euler_angles: list[float] = [0.0, 0.0, 0.0]
    max_euler_angles: list[float] = [0.0, 0.0, 0.0]

    min_state_ratio: list[float] = [
        0.5,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
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
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env: bool = True

    collapse_fixed_joints: bool = True
    per_link_semantic: bool = False
    semantic_id: int = -1


class thin_asset_params(asset_state_params):
    num_assets: int = 0

    asset_folder: str = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/thin"

    collision_mask: int = 1

    min_state_ratio: list[float] = [
        0.3,
        0.05,
        0.05,
        -np.pi,
        -np.pi,
        -np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio: list[float] = [
        0.85,
        0.95,
        0.95,
        np.pi,
        np.pi,
        np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    collapse_fixed_joints: bool = True
    per_link_semantic: bool = False
    semantic_id: int = -1
    color: list[int] | None = [170, 66, 66]


class tree_asset_params(asset_state_params):
    num_assets: int = 1

    asset_folder: str = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/trees"

    collision_mask: int = 1

    min_state_ratio: list[float] = [
        0.1,
        0.1,
        0.0,
        0,
        -np.pi / 6.0,
        -np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio: list[float] = [
        0.9,
        0.9,
        0.0,
        0,
        np.pi / 6.0,
        np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    collapse_fixed_joints: bool = True
    per_link_semantic: bool = True
    keep_in_env: bool = True

    semantic_id: int = -1
    color: list[int] | None = [70, 200, 100]

    semantic_masked_links: dict[str, int] = {}


class object_asset_params(asset_state_params):
    num_assets: int = 5

    asset_folder: str = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"

    min_state_ratio: list[float] = [
        0.30,
        0.05,
        0.05,
        -np.pi,
        -np.pi,
        -np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio: list[float] = [
        0.85,
        0.9,
        0.9,
        np.pi,
        np.pi,
        np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env: bool = False
    per_link_semantic: bool = False
    semantic_id: int = -1


# Re-export wall configs so existing imports from this module continue to work
