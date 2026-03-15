from __future__ import annotations

import math
import os

import torch
from isaacgym import gymapi

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("static_camera_angle_computation")


def read_ablation_flags_and_robot_positions(
    env_manager: object,
) -> tuple[bool, torch.Tensor | None]:
    """Read the orientation-disabled ablation flag and robot positions from global tensor dict.

    Returns:
        A tuple of (disable_flag, robot_positions).
    """
    disable_flag = False
    rp = None
    try:
        gtd = env_manager.global_tensor_dict
        disable_flag = bool(gtd.get("static_camera_randomization/orientation_disabled", False))
        rp = gtd.get("robot_position", None)
    except (KeyError, TypeError, AttributeError):
        pass
    return disable_flag, rp


def resolve_base_camera_position(
    env_manager: object,
) -> tuple[gymapi.Vec3, float | None, dict]:
    """Resolve the base camera position from env vars and global tensor dict.

    Returns:
        A tuple of (base_camera_pos, base_z, gtd) where base_z is None when
        adaptive-per-env mode is active.
    """
    try:
        gtd = env_manager.global_tensor_dict
    except AttributeError:
        gtd = {}
    try:
        base_y = float(
            os.environ.get("SF_STATIC_CAMERA_BASE_Y", gtd.get("static_camera/base_y", -3.0))
        )
    except (ValueError, TypeError):
        base_y = -3.0

    try:
        base_z_env = os.environ.get("SF_STATIC_CAMERA_BASE_Z", None)
        if base_z_env is None:
            base_z_env = gtd.get("static_camera/base_z", 1.5)
        if isinstance(base_z_env, str) and base_z_env.strip().lower() == "adaptive":
            base_z: float | None = None
        else:
            base_z = float(base_z_env)
    except (ValueError, TypeError):
        base_z = 1.5

    base_z_for_vec = 1.5 if base_z is None else float(base_z)
    base_camera_pos = gymapi.Vec3(0.0, base_y, base_z_for_vec)
    return base_camera_pos, base_z, gtd


def resample_jitter_for_env(
    env_idx: int,
    device: str | torch.device,
    min_t: list[float],
    max_t: list[float],
    min_euler: list[float],
    max_euler: list[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Re-sample per-env translation and euler jitter on episode reset.

    Returns:
        A tuple of (trans_jitter, euler_jitter_deg).
    """
    try:
        u = torch.rand(3, device=device)
        tmin = torch.tensor(min_t, device=device, dtype=torch.float32)
        tmax = torch.tensor(max_t, device=device, dtype=torch.float32)
        t = (tmin + u * (tmax - tmin)).tolist()
        v = torch.rand(3, device=device)
        emin = torch.tensor(min_euler, device=device, dtype=torch.float32)
        emax = torch.tensor(max_euler, device=device, dtype=torch.float32)
        e = (emin + v * (emax - emin)).tolist()
        return (float(t[0]), float(t[1]), float(t[2])), (float(e[0]), float(e[1]), float(e[2]))
    except (ValueError, TypeError):
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)


def _get_eval_stretch_info(
    env_manager: object,
) -> tuple[bool, int]:
    """Read eval-stretch settings from global tensor dict and env vars.

    Returns:
        A tuple of (eval_enabled, eval_end_level).
    """
    try:
        gtd = env_manager.global_tensor_dict
    except AttributeError:
        gtd = {}

    eval_en = bool(gtd.get("eval_stretch_enabled", False))
    if not eval_en:
        eval_en = os.environ.get("EVAL_STRETCH_ENABLED", "0").strip() in ("1", "true", "True")

    try:
        from aerial_gym.config.task_config.navigation_task_config_gate import (
            task_config as _tc,
        )

        eval_end = int(
            gtd.get(
                "eval_stretch_end_level",
                _tc.curriculum.eval_stretch_end_level
                if hasattr(_tc.curriculum, "eval_stretch_end_level")
                else 23,
            )
        )
    except (ValueError, TypeError, ImportError):
        eval_end = 23

    return eval_en, eval_end


def compute_yaw_sweep_angle(
    env_idx: int,
    curriculum_level: int,
    sweep_speed_deg: float,
    gtd: dict,
    device: str | torch.device,
    env_manager: object,
    sweep_phase_offsets: list[float],
    sweep_directions: list[float],
) -> tuple[float, float]:
    """Compute yaw sweep angle offset for a single environment.

    Returns:
        A tuple of (angle_offset_degrees, debug_max_range).
    """
    eval_en, eval_end = _get_eval_stretch_info(env_manager)
    start_level = 3
    end_level = int(eval_end) if eval_en else 23

    A_min, A_max = 2.0, 19.0
    if curriculum_level <= start_level:
        A = A_min
    elif curriculum_level >= end_level:
        A = A_max
    else:
        frac = float(curriculum_level - start_level) / max(1.0, float(end_level - start_level))
        A = A_min + frac * (A_max - A_min)

    dt = 1.0 / 60.0
    A0 = 50.0
    comp = A0 / max(A, 1e-6)

    speed_scale = _compute_speed_scale(curriculum_level, env_manager)
    sweep_speed_eff = sweep_speed_deg * speed_scale * comp
    omega = (sweep_speed_eff * 3.14159 / 180.0) * dt

    sim_steps = _extract_sim_steps(gtd, env_idx)

    # Per-env randomized phase and direction
    try:
        if sim_steps == 0:
            rand_phase = float(torch.rand(1, device=device).item())
            sweep_phase_offsets[env_idx] = -math.pi + (2.0 * math.pi * rand_phase)
            dir_flag = int(torch.randint(low=0, high=2, size=(1,), device=device).item())
            sweep_directions[env_idx] = 1.0 if dir_flag == 1 else -1.0
        phi0 = sweep_phase_offsets[env_idx]
        direction = sweep_directions[env_idx]
    except (ValueError, TypeError):
        phi0 = 0.0
        direction = 1.0

    angle_offset_degrees = A * math.sin(direction * (omega * sim_steps) + phi0)
    return angle_offset_degrees, A


def _compute_speed_scale(curriculum_level: int, env_manager: object) -> float:
    """Compute curriculum-dependent sweep speed scale factor."""
    try:
        from aerial_gym.config.task_config.navigation_task_config_gate import (
            task_config as _tc2,
        )

        eval_en, _ = _get_eval_stretch_info(env_manager)
        min_lvl = int(_tc2.curriculum.min_level)
        max_lvl_cfg = int(_tc2.curriculum.max_level)
        max_lvl_eval = int(
            _tc2.curriculum.eval_stretch_end_level
            if hasattr(_tc2.curriculum, "eval_stretch_end_level")
            else max_lvl_cfg
        )
        max_lvl = max_lvl_eval if eval_en else max_lvl_cfg
        level_clamped = max(min(curriculum_level, max_lvl), min_lvl)
        denom = max(1, max_lvl - min_lvl)
        level_frac = float(level_clamped - min_lvl) / float(denom)
        return 1.0 + level_frac
    except (ValueError, TypeError, ImportError):
        return 1.0


def _extract_sim_steps(gtd: dict, env_idx: int) -> int:
    """Extract sim_steps from global tensor dict, handling both tensor and int."""
    try:
        steps_obj = gtd.get("sim_steps", 0)
        if isinstance(steps_obj, torch.Tensor):
            if steps_obj.ndim == 0:
                return int(steps_obj.item())
            idx = env_idx if env_idx < steps_obj.shape[0] else 0
            return int(steps_obj[idx].item())
        return int(steps_obj)
    except (ValueError, TypeError):
        return 0


def compute_spawn_aware_angle(
    env_idx: int,
    curriculum_level: int,
    base_angle_range: float,
    base_camera_pos: gymapi.Vec3,
    disable_flag: bool,
    rp: torch.Tensor | None,
    base_y: float,
    device: str | torch.device,
    env_manager: object,
) -> tuple[float, float]:
    """Compute spawn-aware angle when yaw sweep is disabled.

    Returns:
        A tuple of (angle_offset_degrees, debug_max_range).
    """
    max_angle_range = _extend_angle_range_from_spawn(
        base_angle_range, curriculum_level, base_y, env_manager
    )

    if disable_flag or max_angle_range <= 0:
        return 0.0, max_angle_range

    horizontal_fov = 87.0
    half_fov = horizontal_fov * 0.5
    margin = 2.5

    max_angle_range = _extend_angle_range_for_eval(
        max_angle_range, curriculum_level, env_manager
    )

    if rp is not None and env_idx < rp.shape[0]:
        cam_x, cam_y = base_camera_pos.x, base_camera_pos.y
        dx = float(rp[env_idx, 0].item()) - cam_x
        dy = float(rp[env_idx, 1].item()) - cam_y
        theta_r = math.degrees(math.atan2(dx, dy))
        gate_low, gate_high = -half_fov + margin, half_fov - margin
        rob_low = theta_r - (half_fov - margin)
        rob_high = theta_r + (half_fov - margin)
        low = max(gate_low, rob_low, -max_angle_range)
        high = min(gate_high, rob_high, max_angle_range)
        if high > low:
            u = float(torch.rand(1, device=device).item())
            angle_offset_degrees = low + u * (high - low)
        else:
            target = max(min(theta_r, gate_high), gate_low)
            angle_offset_degrees = max(-max_angle_range, min(max_angle_range, target))
    else:
        u = float(torch.rand(1, device=device).item())
        angle_offset_degrees = -max_angle_range + u * (2.0 * max_angle_range)

    return angle_offset_degrees, max_angle_range


def _extend_angle_range_from_spawn(
    base_angle_range: float,
    curriculum_level: int,
    base_y: float,
    env_manager: object,
) -> float:
    """Extend allowable angle range based on spawn ranges from curriculum config."""
    max_angle_range = base_angle_range
    try:
        from aerial_gym.config.task_config.navigation_task_config_gate import (
            task_config as _tc_fix,
        )

        eval_en, eval_end = _get_eval_stretch_info(env_manager)
        eff_level = min(curriculum_level, eval_end) if eval_en else curriculum_level
        sr_fix = _tc_fix.curriculum.get_spawn_ranges(eff_level)
        x_half_fix = float(sr_fix.get("x_half_span_m", 0.5))
        y_center_fix = float(sr_fix.get("y_center_m", -1.5))
        dy_fix = abs(y_center_fix - float(base_y))
        half_fov_fix = 87.0 * 0.5
        margin_fix = 2.5
        alpha_fix = math.degrees(math.atan2(x_half_fix, max(1e-6, dy_fix)))
        sweep_like_max = max(0.0, alpha_fix - half_fov_fix) + margin_fix
        max_angle_range = max(max_angle_range, sweep_like_max)
    except (ValueError, TypeError, ImportError):
        pass
    return max_angle_range


def _extend_angle_range_for_eval(
    max_angle_range: float,
    curriculum_level: int,
    env_manager: object,
) -> float:
    """Extend angle range further under eval-stretch for levels beyond 23."""
    try:
        eval_en, eval_end = _get_eval_stretch_info(env_manager)
        if eval_en and curriculum_level > 23:
            frac = float(min(curriculum_level, eval_end) - 23) / max(
                1.0, float(eval_end - 23)
            )
            max_angle_range = max_angle_range * (1.0 + 0.25 * frac)
    except (ValueError, TypeError):
        pass
    return max_angle_range


def apply_camera_transform(
    env_idx: int,
    angle_offset_degrees: float,
    sweep_enabled: bool,
    disable_flag: bool,
    max_angle_range: float,
    base_camera_pos: gymapi.Vec3,
    base_z: float | None,
    gtd: dict,
    gym: object,
    env_handles: list,
    camera_handles: list,
    current_camera_angles: list[float],
    euler_jitter_deg: list[tuple[float, float, float]],
    trans_jitter: list[tuple[float, float, float]],
    last_camera_pos: list[tuple[float, float, float]],
    last_camera_target: list[tuple[float, float, float]],
    last_angle_deg: list[float],
) -> None:
    """Apply the computed angle to the Isaac Gym camera for a single environment.

    Resolves per-env Z, applies translation/euler jitter, and calls set_camera_location.
    """
    if env_idx < len(current_camera_angles):
        current_camera_angles[env_idx] = angle_offset_degrees

    angle_offset_radians = angle_offset_degrees * (3.14159 / 180.0)

    jitter_roll_deg, jitter_pitch_deg, jitter_yaw_deg = (
        euler_jitter_deg[env_idx]
        if (0 <= env_idx < len(euler_jitter_deg))
        else (0.0, 0.0, 0.0)
    )
    if sweep_enabled or (not disable_flag and max_angle_range > 0):
        jitter_yaw_deg = 0.0
    pitch_rad = jitter_pitch_deg * (3.14159 / 180.0)

    env_base_z = _resolve_env_base_z(base_z, gtd, env_idx)
    base_camera_env_pos = gymapi.Vec3(base_camera_pos.x, base_camera_pos.y, env_base_z)

    jx, jy, jz = (
        trans_jitter[env_idx] if (0 <= env_idx < len(trans_jitter)) else (0.0, 0.0, 0.0)
    )
    base_camera_env_pos = gymapi.Vec3(
        base_camera_env_pos.x + jx, base_camera_env_pos.y + jy, base_camera_env_pos.z + jz
    )

    target_distance = abs(base_camera_env_pos.y)
    yaw_total = angle_offset_radians + (jitter_yaw_deg * (3.14159 / 180.0))
    target_x = base_camera_env_pos.x + target_distance * math.sin(yaw_total)
    target_y = base_camera_env_pos.y + target_distance * math.cos(yaw_total)

    target_z = _resolve_target_z(gtd, env_idx, env_base_z)
    target_z = target_z + math.tan(pitch_rad) * target_distance
    new_target = gymapi.Vec3(target_x, target_y, target_z)

    env_handle = env_handles[env_idx]
    cam_handle = camera_handles[env_idx]
    gym.set_camera_location(cam_handle, env_handle, base_camera_env_pos, new_target)

    last_camera_pos[env_idx] = (
        float(base_camera_env_pos.x),
        float(base_camera_env_pos.y),
        float(base_camera_env_pos.z),
    )
    last_camera_target[env_idx] = (
        float(new_target.x),
        float(new_target.y),
        float(new_target.z),
    )
    last_angle_deg[env_idx] = float(angle_offset_degrees)


def _resolve_env_base_z(
    base_z: float | None, gtd: dict, env_idx: int
) -> float:
    """Resolve per-env base Z from adaptive gate height or fixed value."""
    try:
        if base_z is None:
            gh = gtd.get("gate/center_height_per_env", None)
            return float(gh[env_idx].item()) if gh is not None else 1.5
        return float(base_z)
    except (ValueError, TypeError):
        return 1.5


def _resolve_target_z(gtd: dict, env_idx: int, fallback_z: float) -> float:
    """Resolve target Z from adaptive gate height or fallback."""
    try:
        gh = gtd.get("gate/center_height_per_env", None)
        return float(gh[env_idx].item()) if gh is not None else fallback_z
    except (ValueError, TypeError):
        return fallback_z
