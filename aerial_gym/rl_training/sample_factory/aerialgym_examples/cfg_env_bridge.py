"""Bridge CLI config flags to environment variables for Sample Factory worker processes."""

from __future__ import annotations

import logging
import os

from sample_factory.utils.typing import Config

logger: logging.Logger = logging.getLogger(__name__)


def bridge_cfg_to_env_vars(cfg: Config) -> None:
    """Propagate CLI ablation/feature flags to env vars so worker subprocesses can read them."""
    os.environ["SF_FUSION_MODE"] = str(cfg.fusion)
    os.environ["SF_GATE_PER_FEATURE"] = "1" if int(cfg.gate_per_feature) != 0 else "0"

    _set_bool_env(
        "SF_DISABLE_STATIC_CAMERA_ORIENT_RANDOMIZATION",
        cfg.disable_static_camera_orientation_randomization,
    )
    _set_bool_env("SF_DISABLE_CAMERA_NOISE_RANDOMIZATION", cfg.disable_camera_noise_randomization)
    _set_bool_env(
        "SF_DISABLE_CAMERA_FRAME_DROPOUT_RANDOMIZATION",
        cfg.disable_camera_frame_dropout_randomization,
    )
    _set_bool_env("SF_DISABLE_STATE_NOISE_RANDOMIZATION", cfg.disable_state_noise_randomization)
    _set_bool_env("disable_dynamic_camera_following", cfg.disable_dynamic_camera_following)
    _set_bool_env("SF_ENABLE_STATIC_CAMERA_YAW_SWEEP", cfg.enable_static_camera_yaw_sweep)
    _set_bool_env("SF_STATIC_CAMERA_LOCKED_FOLLOW", cfg.enable_static_camera_locked)
    _set_bool_env(
        "SF_DISABLE_DYNAMIC_FOLLOW_GATE_BLENDING", cfg.disable_dynamic_follow_gate_blending
    )
    _set_bool_env(
        "SF_DISABLE_SPAWN_POSITION_RANDOMIZATION", cfg.disable_spawn_position_randomization
    )
    _set_bool_env(
        "SF_DISABLE_SPAWN_ORIENTATION_RANDOMIZATION", cfg.disable_spawn_orientation_randomization
    )
    _set_bool_env("SF_DISABLE_CURRICULUM_MULTIPLIER", cfg.disable_curriculum_multiplier)
    _set_bool_env("SF_ENABLE_STATIC_CAMERA_ARC_FOLLOW", cfg.enable_static_camera_arc_follow)

    os.environ["SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG"] = str(
        float(cfg.static_camera_yaw_sweep_speed_deg)
    )

    _bridge_per_camera_overrides(cfg)
    _bridge_static_camera_position(cfg)
    _bridge_optional_float_env("SF_STATIC_CAMERA_ARC_RADIUS_M", cfg.static_camera_arc_radius_m)
    _bridge_optional_float_env(
        "SF_DYNAMIC_CAMERA_FOLLOW_OFFSET_Y", cfg.dynamic_camera_follow_y_offset_m
    )

    if cfg.enable_dynamic_camera_following is not None:
        _set_bool_env("enable_dynamic_camera_following", cfg.enable_dynamic_camera_following)

    _bridge_curriculum_flags(cfg)


def _set_bool_env(key: str, value: bool) -> None:
    os.environ[key] = "true" if value else "false"


def _bridge_per_camera_overrides(cfg: Config) -> None:
    """Propagate per-camera noise/dropout override flags."""
    if cfg.disable_drone_camera_noise_randomization is not None:
        _set_bool_env(
            "SF_DISABLE_DRONE_CAMERA_NOISE_RANDOMIZATION",
            cfg.disable_drone_camera_noise_randomization,
        )
    if cfg.disable_static_camera_noise_randomization is not None:
        _set_bool_env(
            "SF_DISABLE_STATIC_CAMERA_NOISE_RANDOMIZATION",
            cfg.disable_static_camera_noise_randomization,
        )
    if cfg.disable_drone_camera_frame_dropout is not None:
        _set_bool_env(
            "SF_DISABLE_DRONE_CAMERA_FRAME_DROPOUT",
            cfg.disable_drone_camera_frame_dropout,
        )
    if cfg.disable_static_camera_frame_dropout is not None:
        _set_bool_env(
            "SF_DISABLE_STATIC_CAMERA_FRAME_DROPOUT",
            cfg.disable_static_camera_frame_dropout,
        )


def _bridge_static_camera_position(cfg: Config) -> None:
    """Propagate static camera base position overrides."""
    if cfg.static_camera_base_y is not None:
        os.environ["SF_STATIC_CAMERA_BASE_Y"] = str(float(cfg.static_camera_base_y))
    if cfg.static_camera_base_z is not None:
        val = cfg.static_camera_base_z
        if isinstance(val, str) and val.lower() == "adaptive":
            os.environ["SF_STATIC_CAMERA_BASE_Z"] = "adaptive"
        else:
            os.environ["SF_STATIC_CAMERA_BASE_Z"] = str(float(val))


def _bridge_optional_float_env(key: str, value: float | None) -> None:
    if value is not None:
        os.environ[key] = str(float(value))


def _bridge_curriculum_flags(cfg: Config) -> None:
    """Propagate curriculum override flags."""
    if cfg.force_curriculum_level is not None:
        lvl_str = str(cfg.force_curriculum_level).strip().lower()
        if lvl_str and lvl_str != "none":
            os.environ["SF_FORCE_CURRICULUM_LEVEL"] = str(int(lvl_str))
        else:
            os.environ.pop("SF_FORCE_CURRICULUM_LEVEL", None)

    try:
        # evaluation attribute only exists on SF Config when parse_aerialgym_cfg(evaluation=True)
        is_eval = bool(cfg.evaluation) if hasattr(cfg, "evaluation") else False
        if not is_eval:
            min_lvl_override = cfg.min_curriculum_level
            if min_lvl_override is not None:
                min_lvl = int(min_lvl_override)
                max_cap = cfg.max_curriculum_level
                if max_cap is not None:
                    os.environ["SF_MAX_CURRICULUM_LEVEL"] = str(int(max_cap))
                os.environ["SF_MIN_CURRICULUM_LEVEL"] = str(min_lvl)
    except (ValueError, TypeError) as e:
        logger.warning("Failed to bridge curriculum level flags: %s", e)
