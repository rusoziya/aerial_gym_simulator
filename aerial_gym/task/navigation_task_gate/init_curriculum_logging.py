from __future__ import annotations

import os

from aerial_gym.utils.env_flag_utils import read_env_bool
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("navigation_task_gate_init")


def log_initial_curriculum_state(
    task: object,
    fixed_assets_visible: int,
    obstacles_behind_gate: int,
    total_obstacles_in_env: int,
) -> None:
    """Log a detailed summary of the initial curriculum configuration.

    Extracted from ``_init_curriculum`` to keep that method focused on state setup.
    """
    logger.info(f"INITIAL CURRICULUM (Level {task.curriculum_level}):")
    logger.info(
        f"   1. OBSTACLES: {obstacles_behind_gate} behind gate (total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)"
    )

    _log_spawn_ranges(task)
    _log_static_camera_yaw_sweep(task)
    _log_camera_angle(task)
    _log_camera_noise(task)
    _log_camera_frame_dropout(task)
    _log_state_noise(task)

    logger.info(
        f"   8. ASSET MANAGER: Updated both obs_dict and global_tensor_dict with count {total_obstacles_in_env}"
    )

    _log_curriculum_multiplier(task)
    _log_progress_and_evaluation(task)

    task._curriculum.log_curriculum_update(
        f"[INIT] Multi-aspect curriculum initialized at level {task.curriculum_level}"
    )


def _log_spawn_ranges(task: object) -> None:
    """Log spawn range configuration (item 2)."""
    try:
        baseline_level = int(task.task_config.curriculum.min_level)
        gtd = task.sim_env.global_tensor_dict
        pos_dis = bool(gtd.get("spawn_randomization/position_disabled", False))
        yaw_dis = bool(gtd.get("spawn_randomization/orientation_disabled", False))

        sr_active = task.task_config.curriculum.get_spawn_ranges(task.curriculum_level)
        sr_base = task.task_config.curriculum.get_spawn_ranges(baseline_level)

        sr_use = {
            "x_half_span_m": sr_base["x_half_span_m"] if pos_dis else sr_active["x_half_span_m"],
            "y_center_m": sr_base["y_center_m"] if pos_dis else sr_active["y_center_m"],
            "y_half_span_m": (sr_base["y_half_span_m"] if pos_dis else sr_active["y_half_span_m"]),
            "z_center_m": sr_base["z_center_m"] if pos_dis else sr_active["z_center_m"],
            "z_half_span_m": (sr_base["z_half_span_m"] if pos_dis else sr_active["z_half_span_m"]),
            "yaw_abs_rad": sr_base["yaw_abs_rad"] if yaw_dis else sr_active["yaw_abs_rad"],
        }

        if pos_dis or yaw_dis:
            status_pos = "DISABLED" if pos_dis else "ENABLED"
            status_yaw = "DISABLED" if yaw_dis else "ENABLED"
            logger.info(
                f"   2. SPAWN RANDOMIZATION: position={status_pos}, orientation={status_yaw}"
            )
        logger.info(
            f"   2. SPAWN: X∈[{(-sr_use['x_half_span_m']):.1f}, {(+sr_use['x_half_span_m']):.1f}] m, "
            f"Y∈[{(sr_use['y_center_m'] - sr_use['y_half_span_m']):.1f}, {(sr_use['y_center_m'] + sr_use['y_half_span_m']):.1f}] m, "
            f"Z∈[{(sr_use['z_center_m'] - sr_use['z_half_span_m']):.1f}, {(sr_use['z_center_m'] + sr_use['z_half_span_m']):.1f}] m; yaw ±{(sr_use['yaw_abs_rad'] * 57.2958):.1f}°"
        )
    except (ValueError, TypeError) as e:
        logger.info(f"   2. SPAWN: (fallback) Using fixed LMF2 config due to: {e}")


def _log_static_camera_yaw_sweep(task: object) -> None:
    """Log static camera yaw sweep status (item 3) and base position."""
    try:
        yaw_enabled = read_env_bool("SF_ENABLE_STATIC_CAMERA_YAW_SWEEP", default=False)
        yaw_speed = float(os.environ.get("SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG", "10.0"))
    except (ValueError, TypeError):
        yaw_enabled = False
        yaw_speed = 10.0

    try:
        cam_orient_disabled = bool(
            task.sim_env.global_tensor_dict.get(
                "static_camera_randomization/orientation_disabled", False
            )
        )
    except (KeyError, TypeError):
        cam_orient_disabled = False

    dynamic_effective = _get_dynamic_camera_effective(task)

    if yaw_enabled:
        sweep_note = "IGNORED (dynamic camera active)" if dynamic_effective else "ACTIVE"
    else:
        sweep_note = "N/A"

    logger.info(
        f"   3. STATIC CAMERA YAW SWEEP: {'ENABLED' if yaw_enabled else 'DISABLED'} (speed={yaw_speed:.1f} deg/s) — effective: {sweep_note}; orientation_rand={'DISABLED' if cam_orient_disabled else 'ENABLED'}"
    )

    try:
        base_y = float(os.environ.get("SF_STATIC_CAMERA_BASE_Y", -3.0))
        base_z = float(os.environ.get("SF_STATIC_CAMERA_BASE_Z", 1.5))
    except (ValueError, TypeError):
        base_y, base_z = -3.0, 1.5
    logger.info(f"      ↳ static camera base: Y={base_y:.2f} m, Z={base_z:.2f} m")


def _get_dynamic_camera_effective(task: object) -> bool:
    """Return whether dynamic camera following is effectively active."""
    try:
        dyn_cfg = task.task_config.curriculum.enable_dynamic_camera_following
        dyn_dis = bool(
            task.sim_env.global_tensor_dict.get("dynamic_camera_following/disabled", False)
        )
        return bool(dyn_cfg and not dyn_dis)
    except (KeyError, TypeError):
        return False


def _log_camera_angle(task: object) -> None:
    """Log camera angle configuration (item 4)."""
    yaw_enabled = read_env_bool("SF_ENABLE_STATIC_CAMERA_YAW_SWEEP", default=False)

    try:
        cam_orient_disabled = bool(
            task.sim_env.global_tensor_dict.get(
                "static_camera_randomization/orientation_disabled", False
            )
        )
    except (KeyError, TypeError):
        cam_orient_disabled = False

    dynamic_effective = _get_dynamic_camera_effective(task)

    if yaw_enabled and not dynamic_effective:
        logger.info("   4. CAMERA ANGLE: overridden by yaw sweep")
    elif dynamic_effective:
        logger.info("   4. CAMERA ANGLE: suppressed (dynamic camera following active)")
    elif cam_orient_disabled:
        logger.info("   4. CAMERA ANGLE: randomization DISABLED, fixed at 0.0°")
    else:
        logger.info(
            f"   4. CAMERA ANGLE: ±{task.max_camera_angle:.1f}deg max range (randomized per episode reset, fixed during episode)"
        )


def _log_camera_noise(task: object) -> None:
    """Log camera noise progression (item 5)."""
    initial_camera_gaussian_std, initial_camera_dropout_rate = (
        task.task_config.curriculum.get_camera_noise(task.curriculum_level)
    )
    logger.info(
        f"   5. CAMERA NOISE: Gaussian STD={initial_camera_gaussian_std:.4f}, Dropout={initial_camera_dropout_rate * 100:.1f}% (both drone & static)"
    )


def _log_camera_frame_dropout(task: object) -> None:
    """Log camera frame dropout configuration (item 6)."""
    fd = task.task_config.curriculum.get_camera_frame_dropout(task.curriculum_level)
    logger.info(
        f"   6. CAMERA FRAME DROPOUT: drone_total={fd['drone_total'] * 100:.1f}% (freeze {fd['drone_freeze'] * 100:.1f}%, blank {fd['drone_blank'] * 100:.1f}%), static_total={fd['static_total'] * 100:.1f}% (freeze {fd['static_freeze'] * 100:.1f}%, blank {fd['static_blank'] * 100:.1f}%)"
    )


def _log_state_noise(task: object) -> None:
    """Log state noise configuration (item 7)."""
    state_noise_disabled = False
    try:
        state_noise_disabled = bool(
            task.sim_env.global_tensor_dict.get("state_randomization/noise_disabled", False)
        )
    except (KeyError, TypeError):
        state_noise_disabled = bool(task.disable_state_noise_randomization)

    if task.task_config.curriculum.enable_state_noise and not state_noise_disabled:
        sn = task.task_config.curriculum.get_state_noise(task.curriculum_level)
        logger.info(
            f"   7. STATE NOISE: drone_pos_std={sn['drone_pos_std_m']:.4f} m, drone_orient_std={sn['drone_orient_std_rad'] * 57.2958:.3f} deg, "
            f"static_pos_std={sn['static_pos_std_m']:.4f} m, static_orient_std={sn['static_orient_std_rad'] * 57.2958:.3f} deg"
        )
    else:
        logger.info("   7. STATE NOISE: disabled")


def _log_curriculum_multiplier(task: object) -> None:
    """Log curriculum multiplier status (item 9)."""
    cm_disabled = read_env_bool(
        "SF_DISABLE_CURRICULUM_MULTIPLIER", task.task_config.disable_curriculum_multiplier
    )
    if not cm_disabled:
        cm_disabled = bool(task.task_config.disable_curriculum_multiplier)
    try:
        frac_current = (task.curriculum_level - task.task_config.curriculum.min_level) / (
            task.task_config.curriculum.max_level - task.task_config.curriculum.min_level
        )
    except (ZeroDivisionError, AttributeError, TypeError):
        frac_current = 0.0
    frac_eff = 0.0 if cm_disabled else float(frac_current)
    factor = 1.0 + 0.5 * frac_eff
    logger.info(
        f"   9. CURRICULUM MULTIPLIER: {'DISABLED' if cm_disabled else 'ENABLED'} (factor={factor:.3f})"
    )


def _log_progress_and_evaluation(task: object) -> None:
    """Log progress fraction and evaluation settings (items 8/9)."""
    logger.info(
        f"   8. PROGRESS: {task.curriculum_progress_fraction:.3f} (level {task.curriculum_level}/{task.task_config.curriculum.max_level})"
    )
    logger.info(
        f"   9. EVALUATION: Check every {task.task_config.curriculum.check_after_log_instances} instances (success rate threshold: {task.task_config.curriculum.success_rate_for_increase:.3f})"
    )
