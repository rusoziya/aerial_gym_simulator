from __future__ import annotations

from aerial_gym.config.asset_config.env_asset_config import (
    WALL_SEMANTIC_ID,
    back_wall,
    bottom_wall,
    front_wall,
    left_wall,
    right_wall,
    top_wall,
)


class boundary_left_wall(left_wall):
    collision_mask = 0  # Enable collision detection
    semantic_id = WALL_SEMANTIC_ID
    body_semantic_label = WALL_SEMANTIC_ID
    link_semantic_label = WALL_SEMANTIC_ID
    per_link_semantic = True
    place_force_sensor = True
    color = [100, 200, 210]  # MATCH ORIGINAL: Teal walls (was [150, 150, 150])
    keep_in_env = True  # Keep in environment (reverted from False)

    # MATCH ORIGINAL: Full height walls like original environment
    min_state_ratio = [
        0.5,
        1.0,
        0.5,  # MATCH ORIGINAL: Position at Z=0.5 (was 0.4), full height
        0.0,
        0.0,
        0.0,  # Orientation
        1.0,  # MATCH ORIGINAL: Full scale (was 0.6)
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # Velocities
    ]
    max_state_ratio = [
        0.5,
        1.0,
        0.5,  # Same position
        0.0,
        0.0,
        0.0,  # Same orientation
        1.0,  # Same full scale
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # Velocities
    ]


class boundary_right_wall(right_wall):
    collision_mask = 0  # Enable collision detection
    semantic_id = WALL_SEMANTIC_ID
    body_semantic_label = WALL_SEMANTIC_ID
    link_semantic_label = WALL_SEMANTIC_ID
    per_link_semantic = True
    place_force_sensor = True
    color = [100, 200, 210]  # MATCH ORIGINAL: Teal walls (was [150, 150, 150])
    keep_in_env = True  # Keep in environment (reverted from False)

    # MATCH ORIGINAL: Full height walls like original environment
    min_state_ratio = [
        0.5,
        0.0,
        0.5,  # MATCH ORIGINAL: Position at Z=0.5 (was 0.4), full height
        0.0,
        0.0,
        0.0,  # Orientation
        1.0,  # MATCH ORIGINAL: Full scale (was 0.6)
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # Velocities
    ]
    max_state_ratio = [
        0.5,
        0.0,
        0.5,  # Same position
        0.0,
        0.0,
        0.0,  # Same orientation
        1.0,  # Same full scale
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # Velocities
    ]


class boundary_front_wall(front_wall):
    collision_mask = 0  # Enable collision detection
    semantic_id = WALL_SEMANTIC_ID
    body_semantic_label = WALL_SEMANTIC_ID
    link_semantic_label = WALL_SEMANTIC_ID
    per_link_semantic = True
    place_force_sensor = True
    color = [100, 200, 210]  # MATCH ORIGINAL: Teal walls (was [150, 150, 150])
    keep_in_env = True  # Keep in environment (reverted from False)

    # MATCH ORIGINAL: Full height walls like original environment
    min_state_ratio = [
        1.0,
        0.5,
        0.5,  # MATCH ORIGINAL: Position at Z=0.5 (was 0.4), full height
        0.0,
        0.0,
        0.0,  # Orientation
        1.0,  # MATCH ORIGINAL: Full scale (was 0.6)
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # Velocities
    ]
    max_state_ratio = [
        1.0,
        0.5,
        0.5,  # Same position
        0.0,
        0.0,
        0.0,  # Same orientation
        1.0,  # Same full scale
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # Velocities
    ]


class boundary_back_wall(back_wall):
    collision_mask = 0  # Enable collision detection
    semantic_id = WALL_SEMANTIC_ID
    body_semantic_label = WALL_SEMANTIC_ID
    link_semantic_label = WALL_SEMANTIC_ID
    per_link_semantic = True
    place_force_sensor = True
    color = [100, 200, 210]  # MATCH ORIGINAL: Teal walls (was [150, 150, 150])
    keep_in_env = True  # Keep in environment (reverted from False)

    # MATCH ORIGINAL: Full height walls like original environment
    min_state_ratio = [
        0.0,
        0.5,
        0.5,  # MATCH ORIGINAL: Position at Z=0.5 (was 0.4), full height
        0.0,
        0.0,
        0.0,  # Orientation
        1.0,  # MATCH ORIGINAL: Full scale (was 0.6)
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # Velocities
    ]
    max_state_ratio = [
        0.0,
        0.5,
        0.5,  # Same position
        0.0,
        0.0,
        0.0,  # Same orientation
        1.0,  # Same full scale
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # Velocities
    ]


class boundary_bottom_wall(bottom_wall):
    collision_mask = 0  # Enable collision detection (ground plane)
    semantic_id = WALL_SEMANTIC_ID
    body_semantic_label = WALL_SEMANTIC_ID
    link_semantic_label = WALL_SEMANTIC_ID
    per_link_semantic = True
    place_force_sensor = True
    color = [100, 150, 150]  # MATCH ORIGINAL: Darker teal ground (was [100, 80, 60])
    keep_in_env = True  # Keep in environment (reverted from False)


class boundary_top_wall(top_wall):
    collision_mask = 0  # Enable collision detection
    semantic_id = WALL_SEMANTIC_ID
    body_semantic_label = WALL_SEMANTIC_ID
    link_semantic_label = WALL_SEMANTIC_ID
    per_link_semantic = True
    place_force_sensor = True
    color = [100, 200, 210]  # MATCH ORIGINAL: Teal ceiling (was [120, 120, 120])
    keep_in_env = True  # Keep in environment (reverted from False)

    # MATCH ORIGINAL: Keep top wall at full height like original environment
    min_state_ratio = [
        0.5,
        0.5,
        1.0,  # MATCH ORIGINAL: Position at Z=1.0 (full height)
        0.0,
        0.0,
        0.0,  # Orientation
        1.0,  # MATCH ORIGINAL: Full scale
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # Velocities
    ]
    max_state_ratio = [
        0.5,
        0.5,
        1.0,  # Same position
        0.0,
        0.0,
        0.0,  # Same orientation
        1.0,  # Same full scale
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # Velocities
    ]
