"""Episode statistics injection for W&B logging in gate navigation."""

from __future__ import annotations

import math
import os

import torch
from torch import Tensor

VERBOSE = os.environ.get("TRAIN_VERBOSE", "false").lower() == "true"


class TrajectoryAggregator:
    """Running episode-level aggregates for trajectory quality metrics."""

    def __init__(self) -> None:
        self.path_efficiency_sum: float = 0.0
        self.path_efficiency_count: int = 0
        self.min_gate_distance_sum: float = 0.0
        self.min_gate_distance_count: int = 0
        self.time_to_gate_sum: float = 0.0
        self.time_to_gate_count: int = 0
        self.center_offset_sum: float = 0.0
        self.center_offset_count: int = 0
        self.height_offset_sum: float = 0.0
        self.height_offset_count: int = 0
        self.episodes_total: int = 0
        self.episodes_crossed: int = 0

    def update(self, reset_ids: list[int], task: object) -> None:
        """Accumulate per-env trajectory metrics for environments that just reset."""
        per_env = task._last_traj_metrics_per_env if task is not None else None
        if not isinstance(per_env, dict):
            return
        crossed_mask = per_env.get("crossed", None)
        for eid in reset_ids:
            pe = float(per_env["path_efficiency"][eid].item())
            mgd = float(per_env["min_gate_distance"][eid].item())
            ttg = float(per_env["time_to_gate_steps"][eid].item())
            co = float(per_env["center_offset_success"][eid].item())
            ho = float(per_env["height_offset_success"][eid].item())
            if math.isfinite(pe):
                self.path_efficiency_sum += pe
                self.path_efficiency_count += 1
            if math.isfinite(mgd):
                self.min_gate_distance_sum += mgd
                self.min_gate_distance_count += 1
            crossed = False
            if isinstance(crossed_mask, torch.Tensor):
                crossed = bool(crossed_mask[eid].item())
            if crossed and math.isfinite(ttg):
                self.time_to_gate_sum += ttg
                self.time_to_gate_count += 1
            if crossed and math.isfinite(co):
                self.center_offset_sum += co
                self.center_offset_count += 1
            if crossed and math.isfinite(ho):
                self.height_offset_sum += ho
                self.height_offset_count += 1
            self.episodes_total += 1
            if crossed:
                self.episodes_crossed += 1

    def _safe_mean(self, total: float, count: int, none_if_zero: bool = False) -> float | None:
        if count <= 0:
            return None if none_if_zero else float("nan")
        return float(total / count)

    def inject_running_means(self, extra: dict[str, float]) -> None:
        extra["path_efficiency_running_mean"] = self._safe_mean(
            self.path_efficiency_sum, self.path_efficiency_count
        )
        extra["min_gate_distance_running_mean"] = self._safe_mean(
            self.min_gate_distance_sum, self.min_gate_distance_count
        )
        extra["time_to_gate_running_mean"] = self._safe_mean(
            self.time_to_gate_sum, self.time_to_gate_count, none_if_zero=True
        )
        extra["center_offset_running_mean"] = self._safe_mean(
            self.center_offset_sum, self.center_offset_count, none_if_zero=True
        )
        extra["height_offset_running_mean"] = self._safe_mean(
            self.height_offset_sum, self.height_offset_count, none_if_zero=True
        )
        extra["gate_pass_rate"] = float(self.episodes_crossed) / float(max(1, self.episodes_total))
        extra["episodes_total"] = float(self.episodes_total)
        extra["episodes_crossed"] = float(self.episodes_crossed)


class CurriculumCounters:
    """Running totals for success/crash/timeout episode outcomes."""

    def __init__(self) -> None:
        self.total_successes: int = 0
        self.total_crashes: int = 0
        self.total_timeouts: int = 0

    def update_and_inject(
        self,
        extra: dict[str, float],
        infos: dict[str, object],
        terminated: Tensor,
        truncated: Tensor,
    ) -> None:
        if not (
            isinstance(infos, dict)
            and "successes" in infos
            and "crashes" in infos
            and "timeouts" in infos
        ):
            return
        ids = (terminated + truncated).nonzero(as_tuple=True)[0]
        if ids.numel() == 0:
            return
        step_successes = int(infos["successes"][ids].sum().item())
        step_crashes = int(infos["crashes"][ids].sum().item())
        step_timeouts = int(infos["timeouts"][ids].sum().item())
        self.total_successes += step_successes
        self.total_crashes += step_crashes
        self.total_timeouts += step_timeouts
        extra["successes"] = float(step_successes)
        extra["crashes"] = float(step_crashes)
        extra["timeouts"] = float(step_timeouts)
        extra["curriculum/total_successes"] = float(self.total_successes)
        extra["curriculum/total_crashes"] = float(self.total_crashes)
        extra["curriculum/total_timeouts"] = float(self.total_timeouts)


def inject_traj_and_level_stats(
    extra: dict[str, float],
    task: object,
    curr_level: float | None,
) -> None:
    """Inject curriculum level and trajectory averages into extra stats."""
    extra["curriculum_level"] = float(curr_level) if curr_level is not None else -1.0
    extra["curriculum_level_minus_1"] = float(curr_level - 1) if curr_level is not None else -1.0
    traj_avg = task._last_traj_metrics_avg if task is not None else None
    if isinstance(traj_avg, dict):
        for k, v in traj_avg.items():
            extra[k] = float(v)
    for k in (
        "path_efficiency",
        "time_to_gate_steps",
        "min_gate_distance",
        "center_offset_success",
        "height_offset_success",
        "target_success_rate",
        "last_position_x",
        "last_position_y",
        "last_position_z",
        "last_center_distance",
    ):
        if k in extra:
            extra[k] = float(extra[k])


def inject_curriculum_current_mirror(
    extra: dict[str, float],
    infos: dict[str, object],
    task: object,
) -> None:
    """Mirror curriculum/current_* so they always show up in W&B."""
    try:
        cur_lvl_tensor = infos.get("curriculum/current_level", None)
        cur_prog_tensor = infos.get("curriculum/current_progress", None)
        if cur_lvl_tensor is not None:
            val = (
                float(cur_lvl_tensor.mean().item())
                if isinstance(cur_lvl_tensor, torch.Tensor)
                else float(cur_lvl_tensor)
            )
            extra["episode_extra_stats/curriculum/current_level"] = val
            del infos["curriculum/current_level"]
        elif task is not None:
            extra["episode_extra_stats/curriculum/current_level"] = float(task.curriculum_level)
        if cur_prog_tensor is not None:
            val = (
                float(cur_prog_tensor.mean().item())
                if isinstance(cur_prog_tensor, torch.Tensor)
                else float(cur_prog_tensor)
            )
            extra["episode_extra_stats/curriculum/current_progress"] = val
            del infos["curriculum/current_progress"]
        elif task is not None:
            extra["episode_extra_stats/curriculum/current_progress"] = float(
                task.curriculum_progress_fraction
            )
    except (ValueError, TypeError):
        pass


def inject_gate_camera_stats(
    extra: dict[str, float],
    infos: dict[str, object],
) -> None:
    """Inject gate/task-specific and camera alignment stats."""
    gate_keys = (
        "gate/passed",
        "gate/distance",
        "gate/alignment",
        "camera/facing_alignment",
        "camera/alignment_angle_deg",
        "camera/alignment_category",
    )
    for key in gate_keys:
        val = infos.get(key, None)
        if val is not None:
            extra[key] = float(val.mean().item()) if isinstance(val, torch.Tensor) else float(val)


def inject_curriculum_snapshot(
    extra: dict[str, float],
    infos: dict[str, object],
    task: object,
) -> None:
    """Inject curriculum snapshot and progression stats."""
    try:
        snapshot_keys = (
            "curriculum/level",
            "curriculum/progress",
            "curriculum/success_rate",
            "curriculum/crash_rate",
            "curriculum/timeout_rate",
            "curriculum/obstacles_behind_gate",
            "curriculum/total_assets",
            "curriculum/max_level_reached",
            "curriculum/camera_gaussian_std",
            "curriculum/camera_dropout_rate",
            "curriculum/camera_frame_dropout_drone_total",
            "curriculum/camera_frame_dropout_static_total",
            "curriculum/camera_frame_freeze_drone",
            "curriculum/camera_frame_blank_drone",
            "curriculum/camera_frame_freeze_static",
            "curriculum/camera_frame_blank_static",
            "curriculum/camera_max_angle",
            "curriculum/camera_current_angle",
            "curriculum/state_noise_drone_pos_std_m",
            "curriculum/state_noise_drone_orient_std_deg",
            "curriculum/state_noise_static_pos_std_m",
            "curriculum/state_noise_static_orient_std_deg",
        )
        mirrored: set[str] = set()
        for key in snapshot_keys:
            val = infos.get(key, None)
            if val is not None:
                extra[key] = (
                    float(val.mean().item()) if isinstance(val, torch.Tensor) else float(val)
                )
                mirrored.add(key)
        if task is not None:
            _inject_snapshot_fallback(extra, task, mirrored)
    except (ValueError, TypeError):
        pass


def _inject_snapshot_fallback(
    extra: dict[str, float],
    task: object,
    mirrored: set[str],
) -> None:
    """Compute fallback curriculum snapshot values from task attributes."""
    if "curriculum/level" not in mirrored:
        extra["curriculum/level"] = float(task.curriculum_level)
    if "curriculum/progress" not in mirrored:
        extra["curriculum/progress"] = float(task.curriculum_progress_fraction)

    _inject_obstacle_counts(extra, task, mirrored)
    _inject_camera_noise_fallback(extra, task, mirrored)
    _inject_frame_dropout_fallback(extra, task, mirrored)
    _inject_camera_angle_fallback(extra, task, mirrored)
    _inject_state_noise_fallback(extra, task, mirrored)


def _inject_obstacle_counts(
    extra: dict[str, float],
    task: object,
    mirrored: set[str],
) -> None:
    try:
        cur_lvl_val = int(task.curriculum_level)
        curri = task.task_config.curriculum
        if curri is not None:
            obg = int(curri.get_obstacle_count_behind_gate(cur_lvl_val))
        else:
            obg = 0
    except (ValueError, TypeError):
        obg = 0
        cur_lvl_val = 0
    if "curriculum/obstacles_behind_gate" not in mirrored:
        extra["curriculum/obstacles_behind_gate"] = float(obg)
    fixed_assets_visible = 1 + 6
    if "curriculum/total_assets" not in mirrored:
        extra["curriculum/total_assets"] = float(fixed_assets_visible + obg)
    if "curriculum/max_level_reached" not in mirrored:
        extra["curriculum/max_level_reached"] = float(task.max_curriculum_level_reached)


def _inject_camera_noise_fallback(
    extra: dict[str, float],
    task: object,
    mirrored: set[str],
) -> None:
    try:
        curri = task.task_config.curriculum
        cur_lvl_val = int(task.curriculum_level)
        if curri is not None:
            cstd, cdrop = curri.get_camera_noise(cur_lvl_val)
        else:
            cstd, cdrop = 0.0, 0.0
    except (AttributeError, ValueError, TypeError):
        cstd, cdrop = 0.0, 0.0
    if "curriculum/camera_gaussian_std" not in mirrored:
        extra["curriculum/camera_gaussian_std"] = float(cstd)
    if "curriculum/camera_dropout_rate" not in mirrored:
        extra["curriculum/camera_dropout_rate"] = float(cdrop)


def _inject_frame_dropout_fallback(
    extra: dict[str, float],
    task: object,
    mirrored: set[str],
) -> None:
    try:
        curri = task.task_config.curriculum
        cur_lvl_val = int(task.curriculum_level)
        if curri is not None:
            fd = curri.get_camera_frame_dropout(cur_lvl_val)
        else:
            fd = _default_frame_dropout()
    except (AttributeError, ValueError, TypeError):
        fd = _default_frame_dropout()
    mapping = {
        "curriculum/camera_frame_dropout_drone_total": "drone_total",
        "curriculum/camera_frame_dropout_static_total": "static_total",
        "curriculum/camera_frame_freeze_drone": "drone_freeze",
        "curriculum/camera_frame_blank_drone": "drone_blank",
        "curriculum/camera_frame_freeze_static": "static_freeze",
        "curriculum/camera_frame_blank_static": "static_blank",
    }
    for curric_key, fd_key in mapping.items():
        if curric_key not in mirrored:
            extra[curric_key] = float(fd.get(fd_key, 0.0))


def _default_frame_dropout() -> dict[str, float]:
    return {
        "drone_total": 0.0,
        "static_total": 0.0,
        "drone_freeze": 0.0,
        "drone_blank": 0.0,
        "static_freeze": 0.0,
        "static_blank": 0.0,
    }


def _inject_camera_angle_fallback(
    extra: dict[str, float],
    task: object,
    mirrored: set[str],
) -> None:
    try:
        curri = task.task_config.curriculum
        cur_lvl_val = int(task.curriculum_level)
        max_angle = task.max_camera_angle
        if max_angle is None and curri is not None:
            max_angle, _, _ = curri.get_static_camera_difficulty(cur_lvl_val)
    except (AttributeError, ValueError, TypeError):
        max_angle = 0.0
    if "curriculum/camera_max_angle" not in mirrored:
        extra["curriculum/camera_max_angle"] = float(max_angle if max_angle is not None else 0.0)
    try:
        cur_angle = 0.0
        scm = task.static_camera_manager
        if scm is not None and scm.current_camera_angles:
            cur_angle = float(scm.current_camera_angles[0])
    except (ValueError, TypeError):
        cur_angle = 0.0
    if "curriculum/camera_current_angle" not in mirrored:
        extra["curriculum/camera_current_angle"] = float(cur_angle)


def _inject_state_noise_fallback(
    extra: dict[str, float],
    task: object,
    mirrored: set[str],
) -> None:
    try:
        curri = task.task_config.curriculum
        cur_lvl_val = int(task.curriculum_level)
        sn = None
        if curri is not None and curri.enable_state_noise:
            sn = curri.get_state_noise(cur_lvl_val)
    except (AttributeError, ValueError, TypeError):
        sn = None
    if sn is not None:
        extra.setdefault(
            "curriculum/state_noise_drone_pos_std_m", float(sn.get("drone_pos_std_m", 0.0))
        )
        extra.setdefault(
            "curriculum/state_noise_drone_orient_std_deg",
            float(sn.get("drone_orient_std_rad", 0.0) * 57.2958),
        )
        extra.setdefault(
            "curriculum/state_noise_static_pos_std_m", float(sn.get("static_pos_std_m", 0.0))
        )
        extra.setdefault(
            "curriculum/state_noise_static_orient_std_deg",
            float(sn.get("static_orient_std_rad", 0.0) * 57.2958),
        )

    # Success/crash/timeout rates from CurriculumCounters are injected by the caller
