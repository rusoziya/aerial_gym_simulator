from __future__ import annotations

from aerial_gym.config.asset_config.dynamic_env_object_config import asset_state_params
from aerial_gym.config.asset_config.wall_config_factory import (
    BACK_WALL_SEMANTIC_ID,
    BOTTOM_WALL_SEMANTIC_ID,
    FRONT_WALL_SEMANTIC_ID,
    LEFT_WALL_SEMANTIC_ID,
    RIGHT_WALL_SEMANTIC_ID,
    TOP_WALL_SEMANTIC_ID,
    make_wall_classes,
)

left_wall, right_wall, top_wall, bottom_wall, front_wall, back_wall = make_wall_classes(
    asset_state_params
)

__all__ = [
    "FRONT_WALL_SEMANTIC_ID",
    "BACK_WALL_SEMANTIC_ID",
    "LEFT_WALL_SEMANTIC_ID",
    "RIGHT_WALL_SEMANTIC_ID",
    "BOTTOM_WALL_SEMANTIC_ID",
    "TOP_WALL_SEMANTIC_ID",
    "left_wall",
    "right_wall",
    "top_wall",
    "bottom_wall",
    "front_wall",
    "back_wall",
]
