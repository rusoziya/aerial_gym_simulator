from __future__ import annotations

import os

import torch

RAD_TO_DEG: float = 57.2958


def get_global_tensor_dict(task: object) -> dict[str, object]:
    """Return the global tensor dict from the task's sim environment."""
    return task.sim_env.global_tensor_dict


def get_obs_disabled(task: object) -> bool:
    """Check whether obstacle randomization is disabled."""
    return bool(get_global_tensor_dict(task).get("obstacles_randomization/disabled", False))


def get_fixed_assets_visible() -> int:
    """Return the count of always-visible fixed assets (gates + walls + robot)."""
    visible_gates = 1
    walls = 6
    robot = 1
    return visible_gates + walls + robot


def get_yaw_sweep_state() -> tuple[bool, float]:
    """Return (enabled, speed_deg_per_sec) for static camera yaw sweep from env vars."""
    try:
        enabled = (
            str(os.environ.get("SF_ENABLE_STATIC_CAMERA_YAW_SWEEP", "false")).lower() == "true"
        )
        speed = float(os.environ.get("SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG", "10.0"))
    except (ValueError, TypeError):
        enabled = False
        speed = 10.0
    return enabled, speed


def is_dynamic_effective(task: object) -> bool:
    """Check whether dynamic camera following is effectively enabled."""
    dyn_cfg = task.task_config.curriculum.enable_dynamic_camera_following
    dyn_dis = bool(get_global_tensor_dict(task).get("dynamic_camera_following/disabled", False))
    return bool(dyn_cfg and not dyn_dis)


def get_current_camera_angle(task: object) -> float:
    """Return the current camera angle for env 0, or 0.0 if unavailable."""
    scm = task.static_camera_manager
    if scm.current_camera_angles:
        return scm.current_camera_angles[0]
    return 0.0


def get_spawn_ranges(task: object) -> dict[str, float] | None:
    """Compute effective spawn ranges respecting ablation flags. Returns None on error."""
    try:
        baseline_level = int(task.task_config.curriculum.min_level)
        gtd = get_global_tensor_dict(task)
        pos_dis = bool(gtd.get("spawn_randomization/position_disabled", False))
        yaw_dis = bool(gtd.get("spawn_randomization/orientation_disabled", False))
        sr_active = task.task_config.curriculum.get_spawn_ranges(task.curriculum_level)
        sr_base = task.task_config.curriculum.get_spawn_ranges(baseline_level)
        return {
            "x_half_span_m": (sr_base["x_half_span_m"] if pos_dis else sr_active["x_half_span_m"]),
            "y_center_m": sr_base["y_center_m"] if pos_dis else sr_active["y_center_m"],
            "y_half_span_m": (sr_base["y_half_span_m"] if pos_dis else sr_active["y_half_span_m"]),
            "z_center_m": sr_base["z_center_m"] if pos_dis else sr_active["z_center_m"],
            "z_half_span_m": (sr_base["z_half_span_m"] if pos_dis else sr_active["z_half_span_m"]),
            "yaw_abs_rad": sr_base["yaw_abs_rad"] if yaw_dis else sr_active["yaw_abs_rad"],
            "pos_disabled": pos_dis,
            "yaw_disabled": yaw_dis,
        }
    except (ValueError, TypeError):
        return None


def get_camera_noise_effective(
    task: object,
) -> tuple[float, float, float, float, float, float]:
    """Return (gaussian_std, dropout_rate, eff_drone_std, eff_static_std, eff_drone_drop, eff_static_drop)."""
    gaussian_std, dropout_rate = task.task_config.curriculum.get_camera_noise(task.curriculum_level)
    gtd = get_global_tensor_dict(task)
    cam_noise_disabled = bool(gtd.get("camera_randomization/noise_disabled", False))
    drone_noise_flag = bool(
        gtd.get("camera_randomization/drone_noise_disabled", cam_noise_disabled)
    )
    static_noise_flag = bool(
        gtd.get("camera_randomization/static_noise_disabled", cam_noise_disabled)
    )
    d_std_min, d_drop_min = task.task_config.curriculum.get_camera_noise(3)
    return (
        gaussian_std,
        dropout_rate,
        gaussian_std if not drone_noise_flag else d_std_min,
        gaussian_std if not static_noise_flag else d_std_min,
        dropout_rate if not drone_noise_flag else d_drop_min,
        dropout_rate if not static_noise_flag else d_drop_min,
    )


def get_frame_dropout_effective(
    task: object,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (effective_fd, scheduled_fd) dicts with per-camera frame dropout rates."""
    fd = task.task_config.curriculum.get_camera_frame_dropout(task.curriculum_level)
    gtd = get_global_tensor_dict(task)
    global_fd_dis = bool(gtd.get("camera_randomization/frame_dropout_disabled", False))
    drone_fd_flag = bool(
        gtd.get("camera_randomization/drone_frame_dropout_disabled", global_fd_dis)
    )
    static_fd_flag = bool(
        gtd.get("camera_randomization/static_frame_dropout_disabled", global_fd_dis)
    )
    fd_min = task.task_config.curriculum.get_camera_frame_dropout(3)
    eff = {
        "drone_total": fd["drone_total"] if not drone_fd_flag else fd_min["drone_total"],
        "static_total": fd["static_total"] if not static_fd_flag else fd_min["static_total"],
        "drone_freeze": fd["drone_freeze"] if not drone_fd_flag else fd_min["drone_freeze"],
        "drone_blank": fd["drone_blank"] if not drone_fd_flag else fd_min["drone_blank"],
        "static_freeze": fd["static_freeze"] if not static_fd_flag else fd_min["static_freeze"],
        "static_blank": fd["static_blank"] if not static_fd_flag else fd_min["static_blank"],
    }
    return eff, fd


def to_tensor_scalar(val: object) -> bool:
    """Safely convert a value (possibly a tensor) to bool."""
    if isinstance(val, torch.Tensor):
        return bool(val.item())
    return bool(val)


def compute_min_gate_scale(task: object) -> int:
    """Compute the minimum gate scale percentage based on curriculum level."""
    stretch_enabled = os.environ.get("EVAL_STRETCH_ENABLED", "0").strip() in (
        "1",
        "true",
        "True",
    )
    eval_end = int(
        os.environ.get(
            "EVAL_STRETCH_END_LEVEL",
            str(task.task_config.curriculum.eval_stretch_end_level),
        )
    )
    level = int(task.curriculum_level)
    if level <= 3:
        min_scale = 80
    elif level <= 23:
        frac = (level - 3) / (23 - 3)
        raw = 80 - frac * (80 - 60)
        min_scale = int((int(raw) // 2) * 2)
    elif stretch_enabled:
        if level >= eval_end:
            min_scale = 50
        else:
            extra_frac = (level - 23) / max(1, (eval_end - 23))
            raw = 60 - extra_frac * (60 - 50)
            min_scale = int((int(raw) // 2) * 2)
    else:
        min_scale = 60
    return max(50, min(100, min_scale))


def collect_unlocked_scales(gate_names: list[str], min_scale: int) -> list[int]:
    """Collect and sort gate scales that are at or above the minimum scale threshold."""
    scales: list[int] = []
    for n in gate_names:
        if isinstance(n, str) and "gate_scale_" in n:
            try:
                s = int(n.replace("gate_scale_", ""))
                if s >= min_scale:
                    scales.append(s)
            except (ValueError, TypeError):
                pass
    return sorted(set(scales), reverse=True)
