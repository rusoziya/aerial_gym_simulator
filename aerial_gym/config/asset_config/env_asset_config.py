from __future__ import annotations

from aerial_gym.config.asset_config.base_asset import *

import numpy as np

THIN_SEMANTIC_ID = 1
TREE_SEMANTIC_ID = 2
OBJECT_SEMANTIC_ID = 3
WALL_SEMANTIC_ID = 8


def _make_wall_class(
    name: str,
    urdf_file: str,
    position: list[float],
    has_collision_mask: bool = True,
    has_per_link_semantic: bool = True,
) -> type:
    """Generate a wall config class with the given position and URDF."""
    state_ratio = position + [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    attrs = {
        "num_assets": 1,
        "asset_folder": f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls",
        "file": urdf_file,
        "min_state_ratio": state_ratio,
        "max_state_ratio": state_ratio,
        "keep_in_env": True,
        "collapse_fixed_joints": True,
        "semantic_id": -1,
        "color": [100, 200, 210],
    }
    if has_collision_mask:
        attrs["collision_mask"] = 1
    if has_per_link_semantic:
        attrs["per_link_semantic"] = True
    return type(name, (BaseAssetParams,), attrs)


class EnvObjectConfig:
    class panel_asset_params(BaseAssetParams):
        num_assets = 6

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/panels"

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.3, 0.05, 0.05]  # max position as a ratio of the bounds
        max_position_ratio = [0.85, 0.95, 0.95]  # min position as a ratio of the bounds

        specified_position = [
            -1000.0,
            -1000.0,
            -1000.0,
        ]  # if > -900, use this value instead of randomizing   the ratios

        min_euler_angles = [0.0, 0.0, -np.pi / 3.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, np.pi / 3.0]  # max euler angles

        min_state_ratio = [
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
        max_state_ratio = [
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

        keep_in_env = True

        collapse_fixed_joints = True
        per_link_semantic = True
        semantic_id = -1
        color = [170, 66, 66]

    class thin_asset_params(BaseAssetParams):
        num_assets = 0

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/thin"

        collision_mask = 1  # objects with the same collision mask will not collide

        min_state_ratio = [
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
        max_state_ratio = [
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

        collapse_fixed_joints = True
        semantic_id = THIN_SEMANTIC_ID
        color = [170, 66, 66]

    class tree_asset_params(BaseAssetParams):
        num_assets = 1

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/trees"

        collision_mask = 1  # objects with the same collision mask will not collide

        min_state_ratio = [
            0.2,
            0.05,
            0.05,
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
        max_state_ratio = [
            0.9,
            0.9,
            0.9,
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

        collapse_fixed_joints = True
        per_link_semantic = False

        semantic_id = TREE_SEMANTIC_ID
        color = [70, 200, 100]

        semantic_masked_links = {}

    class object_asset_params(BaseAssetParams):
        num_assets = 2

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"

        min_state_ratio = [
            0.25,
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
        max_state_ratio = [
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

        semantic_id = OBJECT_SEMANTIC_ID

        # color = [80,255,100]

    left_wall = _make_wall_class("left_wall", "left_wall.urdf", [0.5, 1.0, 0.5])
    right_wall = _make_wall_class("right_wall", "right_wall.urdf", [0.5, 0.0, 0.5], has_collision_mask=False, has_per_link_semantic=False)
    top_wall = _make_wall_class("top_wall", "top_wall.urdf", [0.5, 0.5, 1.0])
    bottom_wall = _make_wall_class("bottom_wall", "bottom_wall.urdf", [0.5, 0.5, 0.0])
    front_wall = _make_wall_class("front_wall", "front_wall.urdf", [1.0, 0.5, 0.5])
    back_wall = _make_wall_class("back_wall", "back_wall.urdf", [0.0, 0.5, 0.5], has_per_link_semantic=False)



# Create module-level exports for easier importing
panel_asset_params = EnvObjectConfig.panel_asset_params
thin_asset_params = EnvObjectConfig.thin_asset_params
tree_asset_params = EnvObjectConfig.tree_asset_params
object_asset_params = EnvObjectConfig.object_asset_params
left_wall = EnvObjectConfig.left_wall
right_wall = EnvObjectConfig.right_wall
top_wall = EnvObjectConfig.top_wall
bottom_wall = EnvObjectConfig.bottom_wall
front_wall = EnvObjectConfig.front_wall
back_wall = EnvObjectConfig.back_wall
