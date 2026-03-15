from __future__ import annotations

from aerial_gym import AERIAL_GYM_DIRECTORY


class BaseAssetParams:
    num_assets: int = 1

    asset_folder: str = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets"
    file: str | None = None

    min_position_ratio: list[float] = [0.5, 0.5, 0.5]
    max_position_ratio: list[float] = [0.5, 0.5, 0.5]

    collision_mask: int = 1

    disable_gravity: bool = False
    replace_cylinder_with_capsule: bool = True
    flip_visual_attachments: bool = True
    density: float = 0.000001
    angular_damping: float = 0.0001
    linear_damping: float = 0.0001
    max_angular_velocity: float = 100.0
    max_linear_velocity: float = 100.0
    armature: float = 0.001

    collapse_fixed_joints: bool = True
    fix_base_link: bool = True
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
