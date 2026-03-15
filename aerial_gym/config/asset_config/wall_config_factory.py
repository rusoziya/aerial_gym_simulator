from __future__ import annotations

from aerial_gym import AERIAL_GYM_DIRECTORY

FRONT_WALL_SEMANTIC_ID = 9
BACK_WALL_SEMANTIC_ID = 10
LEFT_WALL_SEMANTIC_ID = 11
RIGHT_WALL_SEMANTIC_ID = 12
BOTTOM_WALL_SEMANTIC_ID = 13
TOP_WALL_SEMANTIC_ID = 14

_WALL_FOLDER = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"
_ZERO_TAIL = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _state_ratio(x: float, y: float, z: float) -> list[float]:
    return [x, y, z] + _ZERO_TAIL


def make_wall_classes(
    base_class: type,
) -> tuple[type, type, type, type, type, type]:
    """Generate the six boundary-wall config classes from *base_class*.

    Returns (left_wall, right_wall, top_wall, bottom_wall, front_wall, back_wall).
    """

    class left_wall(base_class):
        num_assets = 1
        asset_folder = _WALL_FOLDER
        file = "left_wall.urdf"
        collision_mask = 1
        min_state_ratio = _state_ratio(0.5, 1.0, 0.5)
        max_state_ratio = _state_ratio(0.5, 1.0, 0.5)
        keep_in_env = True
        collapse_fixed_joints = True
        specific_filepath = "cube.urdf"
        per_link_semantic = False
        semantic_id = LEFT_WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class right_wall(base_class):
        num_assets = 1
        asset_folder = _WALL_FOLDER
        file = "right_wall.urdf"
        min_state_ratio = _state_ratio(0.5, 0.0, 0.5)
        max_state_ratio = _state_ratio(0.5, 0.0, 0.5)
        keep_in_env = True
        collapse_fixed_joints = True
        per_link_semantic = False
        specific_filepath = "cube.urdf"
        semantic_id = RIGHT_WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class top_wall(base_class):
        num_assets = 1
        asset_folder = _WALL_FOLDER
        file = "top_wall.urdf"
        collision_mask = 1
        min_state_ratio = _state_ratio(0.5, 0.5, 1.0)
        max_state_ratio = _state_ratio(0.5, 0.5, 1.0)
        keep_in_env = True
        collapse_fixed_joints = True
        specific_filepath = "cube.urdf"
        per_link_semantic = False
        semantic_id = TOP_WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class bottom_wall(base_class):
        num_assets = 1
        asset_folder = _WALL_FOLDER
        file = "bottom_wall.urdf"
        collision_mask = 1
        min_state_ratio = _state_ratio(0.5, 0.5, 0.0)
        max_state_ratio = _state_ratio(0.5, 0.5, 0.0)
        keep_in_env = True
        collapse_fixed_joints = True
        specific_filepath = "cube.urdf"
        per_link_semantic = False
        semantic_id = BOTTOM_WALL_SEMANTIC_ID
        color = [100, 150, 150]

    class front_wall(base_class):
        num_assets = 1
        asset_folder = _WALL_FOLDER
        file = "front_wall.urdf"
        collision_mask = 1
        min_state_ratio = _state_ratio(1.0, 0.5, 0.5)
        max_state_ratio = _state_ratio(1.0, 0.5, 0.5)
        keep_in_env = True
        collapse_fixed_joints = True
        specific_filepath = "cube.urdf"
        per_link_semantic = False
        semantic_id = FRONT_WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class back_wall(base_class):
        num_assets = 1
        asset_folder = _WALL_FOLDER
        file = "back_wall.urdf"
        collision_mask = 1
        min_state_ratio = _state_ratio(0.0, 0.5, 0.5)
        max_state_ratio = _state_ratio(0.0, 0.5, 0.5)
        keep_in_env = True
        collapse_fixed_joints = True
        specific_filepath = "cube.urdf"
        per_link_semantic = False
        semantic_id = BACK_WALL_SEMANTIC_ID
        color = [100, 200, 210]

    return left_wall, right_wall, top_wall, bottom_wall, front_wall, back_wall
