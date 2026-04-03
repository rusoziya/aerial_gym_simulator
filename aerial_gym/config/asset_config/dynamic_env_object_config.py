from __future__ import annotations

import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY
from aerial_gym.config.asset_config.env_object_config import (
    OBJECT_SEMANTIC_ID,
    PANEL_SEMANTIC_ID,
    THIN_SEMANTIC_ID,
    TREE_SEMANTIC_ID,
)
from aerial_gym.config.asset_config.env_object_config import (
    asset_state_params as _static_base,
)

__all__ = [
    "THIN_SEMANTIC_ID",
    "TREE_SEMANTIC_ID",
    "OBJECT_SEMANTIC_ID",
    "PANEL_SEMANTIC_ID",
    "asset_state_params",
    "panel_asset_params",
    "thin_asset_params",
    "tree_asset_params",
    "object_asset_params",
]


class asset_state_params(_static_base):
    """Dynamic variant — gravity disabled, base link unfixed."""

    disable_gravity = True
    fix_base_link = False


class panel_asset_params(asset_state_params):
    num_assets = 3
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/panels"
    collision_mask = 1
    min_position_ratio = [0.3, 0.05, 0.05]
    max_position_ratio = [0.85, 0.95, 0.95]
    specified_position = [-1000.0, -1000.0, -1000.0]
    min_euler_angles = [0.0, 0.0, -np.pi / 3.0]
    max_euler_angles = [0.0, 0.0, np.pi / 3.0]
    min_state_ratio = [
        0.3, 0.05, 0.05, 0.0, 0.0, -np.pi / 3.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]  # fmt: skip
    max_state_ratio = [
        0.85, 0.95, 0.95, 0.0, 0.0, np.pi / 3.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]  # fmt: skip
    keep_in_env = True
    collapse_fixed_joints = True
    per_link_semantic = False
    semantic_id = -1
    color = [170, 66, 66]


class thin_asset_params(asset_state_params):
    num_assets = 0
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/thin"
    collision_mask = 1
    min_state_ratio = [
        0.3, 0.05, 0.05, -np.pi, -np.pi, -np.pi,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]  # fmt: skip
    max_state_ratio = [
        0.85, 0.95, 0.95, np.pi, np.pi, np.pi,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]  # fmt: skip
    collapse_fixed_joints = True
    per_link_semantic = False
    semantic_id = -1
    color = [170, 66, 66]


class tree_asset_params(asset_state_params):
    num_assets = 6
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/trees"
    collision_mask = 1
    min_state_ratio = [
        0.1, 0.1, 0.0, 0, -np.pi / 6.0, -np.pi,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]  # fmt: skip
    max_state_ratio = [
        0.9, 0.9, 0.0, 0, np.pi / 6.0, np.pi,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]  # fmt: skip
    collapse_fixed_joints = True
    per_link_semantic = True
    keep_in_env = True
    semantic_id = -1
    color = [70, 200, 100]
    semantic_masked_links = {}


class object_asset_params(asset_state_params):
    num_assets = 40
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"
    min_state_ratio = [
        0.30, 0.05, 0.05, -np.pi, -np.pi, -np.pi,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]  # fmt: skip
    max_state_ratio = [
        0.85, 0.9, 0.9, np.pi, np.pi, np.pi,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]  # fmt: skip
    keep_in_env = False
    per_link_semantic = False
    semantic_id = -1


# Re-export wall configs so existing imports from this module continue to work
