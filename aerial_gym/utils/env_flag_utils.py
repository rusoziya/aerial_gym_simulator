"""Utilities for reading ablation/configuration flags from environment variables.

The pattern throughout the codebase is:
    1. Try to read an env var (e.g. SF_DISABLE_OBSTACLE_RANDOMIZATION)
    2. Fall back to a config value
    3. Fall back to a default value

These helpers collapse that try-except boilerplate into single-line calls.
"""

import os


def read_env_bool(env_var, config_value=None, default=False):
    """Read a boolean flag from an environment variable with config fallback.

    Args:
        env_var: Environment variable name to check first.
        config_value: Direct config attribute value to use if env var is unset.
        default: Final fallback if both env var and config_value are None.

    Truthy values: 'true', '1', 'yes', 'y' (case-insensitive).
    """
    val = os.getenv(env_var)
    if val is not None:
        return str(val).strip().lower() in ("true", "1", "yes", "y")
    if config_value is not None:
        return bool(config_value)
    return default


def read_env_int(env_var, config_value=None, default=0):
    """Read an integer from an environment variable with config fallback."""
    val = os.getenv(env_var)
    if val is not None:
        try:
            return int(val)
        except (ValueError, TypeError):
            pass
    if config_value is not None:
        return int(config_value)
    return default


def read_env_float(env_var, config_value=None, default=0.0):
    """Read a float from an environment variable with config fallback."""
    val = os.getenv(env_var)
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    if config_value is not None:
        return float(config_value)
    return default


def parse_ablation_flags(task_config):
    """Parse all ablation environment variables and return a flat dict.

    This centralises the ~25 SF_DISABLE_*/SF_ENABLE_*/SF_FIXED_* env vars
    that gate-navigation tasks read at startup.
    """
    tc = task_config
    flags = {}

    # Obstacle randomisation
    flags["obstacle_randomization_disabled"] = read_env_bool(
        "SF_DISABLE_OBSTACLE_RANDOMIZATION", tc.disable_obstacle_randomization
    )
    flags["fixed_obstacles_behind_gate"] = read_env_int(
        "SF_FIXED_OBSTACLES_BEHIND_GATE", tc.fixed_obstacles_behind_gate
    )

    # Gate size randomisation
    flags["gate_size_randomization_disabled"] = read_env_bool(
        "SF_DISABLE_GATE_SIZE_RANDOMIZATION", tc.disable_gate_size_randomization
    )
    flags["fixed_gate_scale_percent"] = read_env_int(
        "SF_FIXED_GATE_SCALE_PERCENT", tc.fixed_gate_scale_percent, default=100
    )

    # Static camera orientation
    flags["static_camera_orient_disabled"] = read_env_bool(
        "SF_DISABLE_STATIC_CAMERA_ORIENT_RANDOMIZATION",
        tc.disable_static_camera_orientation_randomization,
    )

    # Camera noise
    flags["camera_noise_disabled"] = read_env_bool(
        "SF_DISABLE_CAMERA_NOISE_RANDOMIZATION", tc.disable_camera_noise_randomization
    )
    flags["drone_camera_noise_disabled"] = read_env_bool(
        "SF_DISABLE_DRONE_CAMERA_NOISE_RANDOMIZATION"
    )
    flags["static_camera_noise_disabled"] = read_env_bool(
        "SF_DISABLE_STATIC_CAMERA_NOISE_RANDOMIZATION"
    )

    # Camera frame dropout
    flags["camera_frame_dropout_disabled"] = read_env_bool(
        "SF_DISABLE_CAMERA_FRAME_DROPOUT_RANDOMIZATION",
        tc.disable_camera_frame_dropout_randomization,
    )
    flags["drone_frame_dropout_disabled"] = read_env_bool(
        "SF_DISABLE_DRONE_CAMERA_FRAME_DROPOUT"
    )
    flags["static_frame_dropout_disabled"] = read_env_bool(
        "SF_DISABLE_STATIC_CAMERA_FRAME_DROPOUT"
    )

    # Spawn randomisation
    flags["spawn_position_disabled"] = read_env_bool(
        "SF_DISABLE_SPAWN_POSITION_RANDOMIZATION", tc.disable_spawn_position_randomization
    )
    flags["spawn_orientation_disabled"] = read_env_bool(
        "SF_DISABLE_SPAWN_ORIENTATION_RANDOMIZATION", tc.disable_spawn_orientation_randomization
    )

    # State noise
    flags["state_noise_disabled"] = read_env_bool(
        "SF_DISABLE_STATE_NOISE_RANDOMIZATION", tc.disable_state_noise_randomization
    )

    # Dynamic camera following
    flags["dynamic_camera_following_disabled"] = read_env_bool(
        "disable_dynamic_camera_following", tc.disable_dynamic_camera_following
    )
    # Enable override (raw env value, None if unset)
    flags["dynamic_camera_enable_override"] = os.getenv("enable_dynamic_camera_following")

    # Static camera yaw sweep
    flags["yaw_sweep_enabled"] = read_env_bool("SF_ENABLE_STATIC_CAMERA_YAW_SWEEP")
    flags["yaw_sweep_speed_deg"] = read_env_float(
        "SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG", default=10.0
    )

    # Locked follow & arc follow
    flags["locked_follow"] = read_env_bool("SF_STATIC_CAMERA_LOCKED_FOLLOW")
    flags["arc_follow_enabled"] = read_env_bool("SF_ENABLE_STATIC_CAMERA_ARC_FOLLOW")
    flags["arc_follow_radius_m"] = read_env_float(
        "SF_STATIC_CAMERA_ARC_RADIUS_M", default=2.0
    )

    # Dynamic follow tuning
    flags["dynamic_follow_offset_y"] = read_env_float(
        "SF_DYNAMIC_CAMERA_FOLLOW_OFFSET_Y", default=-1.0
    )
    flags["dynamic_follow_gate_blending_disabled"] = read_env_bool(
        "SF_DISABLE_DYNAMIC_FOLLOW_GATE_BLENDING"
    )

    # Curriculum multiplier
    flags["curriculum_multiplier_disabled"] = read_env_bool(
        "SF_DISABLE_CURRICULUM_MULTIPLIER", tc.disable_curriculum_multiplier
    )

    return flags


def apply_ablation_flags_to_tensor_dict(gtd, flags, task_config=None, logger=None):
    """Store parsed ablation flags into the global_tensor_dict.

    Also applies side-effects (task_config overrides) where the original code did so.
    """
    f = flags

    # Gate randomisation
    gtd["gate_randomization/disabled"] = f["gate_size_randomization_disabled"]
    gtd["gate_randomization/fixed_scale_percent"] = f["fixed_gate_scale_percent"]

    # Obstacles
    gtd["obstacles_randomization/disabled"] = f["obstacle_randomization_disabled"]
    gtd["obstacles_randomization/fixed_count"] = max(0, f["fixed_obstacles_behind_gate"])

    # Static camera orientation
    gtd["static_camera_randomization/orientation_disabled"] = f["static_camera_orient_disabled"]

    # Camera noise
    gtd["camera_randomization/noise_disabled"] = f["camera_noise_disabled"]
    if f["drone_camera_noise_disabled"]:
        gtd["camera_randomization/drone_noise_disabled"] = True
    if f["static_camera_noise_disabled"]:
        gtd["camera_randomization/static_noise_disabled"] = True

    # Frame dropout
    gtd["camera_randomization/frame_dropout_disabled"] = f["camera_frame_dropout_disabled"]
    if f["drone_frame_dropout_disabled"]:
        gtd["camera_randomization/drone_frame_dropout_disabled"] = True
    if f["static_frame_dropout_disabled"]:
        gtd["camera_randomization/static_frame_dropout_disabled"] = True

    # Spawn randomisation
    gtd["spawn_randomization/position_disabled"] = f["spawn_position_disabled"]
    gtd["spawn_randomization/orientation_disabled"] = f["spawn_orientation_disabled"]

    # State noise
    gtd["state_randomization/noise_disabled"] = f["state_noise_disabled"]

    # Dynamic camera following
    gtd["dynamic_camera_following/disabled"] = f["dynamic_camera_following_disabled"]

    # Yaw sweep
    gtd["static_camera/yaw_sweep_enabled"] = "true" if f["yaw_sweep_enabled"] else "false"
    gtd["static_camera/yaw_sweep_speed_deg"] = float(f["yaw_sweep_speed_deg"])

    # Locked follow
    gtd["static_camera/locked_follow"] = f["locked_follow"]

    # Dynamic camera enable override
    dyn_enable = f.get("dynamic_camera_enable_override")
    if dyn_enable is not None:
        enable_val = str(dyn_enable).lower() == "true"
        if task_config is not None:
            task_config.curriculum.enable_dynamic_camera_following = enable_val
        gtd["dynamic_camera_following/config_overridden"] = True
        if logger:
            logger.warning(
                f"[Dynamic Camera] Config overridden by command line: "
                f"enable_dynamic_camera_following = {enable_val}"
            )
    else:
        gtd["dynamic_camera_following/config_overridden"] = False

    # Arc follow
    gtd["static_camera/arc_follow_enabled"] = f["arc_follow_enabled"]
    gtd["static_camera/arc_follow_radius_m"] = float(f["arc_follow_radius_m"])

    # Dynamic follow tuning
    gtd["dynamic_camera_following/offset_y_m"] = float(f["dynamic_follow_offset_y"])
    gtd["dynamic_camera_following/disable_gate_blending"] = f[
        "dynamic_follow_gate_blending_disabled"
    ]
