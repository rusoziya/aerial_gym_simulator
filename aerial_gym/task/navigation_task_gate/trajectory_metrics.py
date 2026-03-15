from __future__ import annotations

import torch

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("navigation_task_gate_trajectory_metrics")


def stash_per_env_trajectory_metrics(
    task: object,
    env_ids: torch.Tensor,
    path_eff: torch.Tensor,
    time_to_gate: torch.Tensor,
    min_gate_dist: torch.Tensor,
    center_offset: torch.Tensor,
    height_offset: torch.Tensor,
    last_pos_x: torch.Tensor,
    last_pos_y: torch.Tensor,
    last_pos_z: torch.Tensor,
    last_center_offset_vals: torch.Tensor,
    last_height_offset_vals: torch.Tensor,
) -> None:
    """Stash per-env episode trajectory metrics for worker-side running aggregation."""
    try:
        task._last_traj_metrics_per_env = {
            "path_efficiency": path_eff.detach().clone(),
            "time_to_gate_steps": time_to_gate.detach().clone(),
            "min_gate_distance": min_gate_dist.detach().clone(),
            "center_offset_success": center_offset.detach().clone(),
            "height_offset_success": height_offset.detach().clone(),
            "target_success_flag": task._ep_target_success_flag.detach().clone(),
            "last_position_x": torch.full((task.num_envs,), float("nan"), device=task.device),
            "last_position_y": torch.full((task.num_envs,), float("nan"), device=task.device),
            "last_position_z": torch.full((task.num_envs,), float("nan"), device=task.device),
            "last_center_offset": torch.full((task.num_envs,), float("nan"), device=task.device),
            "last_height_offset": torch.full((task.num_envs,), float("nan"), device=task.device),
            "crossed": task._ep_gate_crossed.detach().clone(),
        }
        task._last_traj_metrics_per_env["last_position_x"][env_ids] = last_pos_x
        task._last_traj_metrics_per_env["last_position_y"][env_ids] = last_pos_y
        task._last_traj_metrics_per_env["last_position_z"][env_ids] = last_pos_z
        task._last_traj_metrics_per_env["last_center_offset"][env_ids] = last_center_offset_vals
        task._last_traj_metrics_per_env["last_height_offset"][env_ids] = last_height_offset_vals
    except (ValueError, TypeError):
        task._last_traj_metrics_per_env = None


def stash_averaged_trajectory_metrics(
    task: object,
    env_ids: torch.Tensor,
    pe_avg: torch.Tensor,
    ttg_avg: torch.Tensor,
    mgd_avg: torch.Tensor,
    co_avg: torch.Tensor,
    ho_avg: torch.Tensor,
    lpx_avg: torch.Tensor,
    lpy_avg: torch.Tensor,
    lpz_avg: torch.Tensor,
    lco_avg: torch.Tensor,
    lho_avg: torch.Tensor,
    overall_success_rate: torch.Tensor,
    target_success_rate: torch.Tensor,
    time_to_gate: torch.Tensor,
) -> None:
    """Stash averaged trajectory metrics for logging."""
    try:
        # Fallback to episode-end offsets if crossing-based offsets are NaN
        try:
            co_val = co_avg
            if torch.isnan(co_val):
                co_val = lco_avg
        except (RuntimeError, TypeError):
            co_val = lco_avg
        try:
            ho_val = ho_avg
            if torch.isnan(ho_val):
                ho_val = lho_avg
        except (RuntimeError, TypeError):
            ho_val = lho_avg
        # Build metrics dict while avoiding undefined time-to-gate when no crossing occurred
        _metrics_avg = {
            "path_efficiency": float(pe_avg.item()),
            "min_gate_distance": float(mgd_avg.item()),
            "center_offset_success": float(co_val.item()),
            "height_offset_success": float(ho_val.item()),
            # Duplicate keys to match existing dashboards
            "center_offset": float(co_val.item()),
            "height_offset": float(ho_val.item()),
            "success_rate": float(overall_success_rate.item()),
            "target_success_rate": float(target_success_rate.item()),
            "last_position_x": float(lpx_avg.item()),
            "last_position_y": float(lpy_avg.item()),
            "last_position_z": float(lpz_avg.item()),
            "last_center_offset": float(lco_avg.item()),
            "last_height_offset": float(lho_avg.item()),
        }
        # Only include time-to-gate (steps/seconds) if any env in this reset batch actually crossed
        try:
            num_crossed = int(torch.isfinite(time_to_gate[env_ids]).sum().item())
        except (ValueError, TypeError):
            num_crossed = 0
        if num_crossed > 0 and not torch.isnan(ttg_avg):
            _metrics_avg["time_to_gate_steps"] = float(ttg_avg.item())
            _metrics_avg["time_to_gate"] = float(ttg_avg.item())
        task._last_traj_metrics_avg = _metrics_avg
    except (ValueError, TypeError):
        task._last_traj_metrics_avg = None


def populate_episode_extra_stats(task: object) -> None:
    """Populate infos['episode_extra_stats'] with trajectory metrics and camera ablation flags."""
    extra = task.infos.get("episode_extra_stats", {})
    if not isinstance(extra, dict):
        extra = {}
    extra.update(task._last_traj_metrics_avg or {})
    # Expose per-camera noise/frame-drop overrides to W&B, mirroring prior style
    gtd = task.sim_env.global_tensor_dict
    cam_noise_global = bool(gtd.get("camera_randomization/noise_disabled", False))
    cam_fd_global = bool(gtd.get("camera_randomization/frame_dropout_disabled", False))
    drone_noise_dis = (
        bool(gtd.get("camera_randomization/drone_noise_disabled", False))
        if "camera_randomization/drone_noise_disabled" in gtd
        else cam_noise_global
    )
    static_noise_dis = (
        bool(gtd.get("camera_randomization/static_noise_disabled", False))
        if "camera_randomization/static_noise_disabled" in gtd
        else cam_noise_global
    )
    drone_fd_dis = (
        bool(gtd.get("camera_randomization/drone_frame_dropout_disabled", False))
        if "camera_randomization/drone_frame_dropout_disabled" in gtd
        else cam_fd_global
    )
    static_fd_dis = (
        bool(gtd.get("camera_randomization/static_frame_dropout_disabled", False))
        if "camera_randomization/static_frame_dropout_disabled" in gtd
        else cam_fd_global
    )
    extra["episode_extra_stats/camera_noise_disabled_drone"] = float(drone_noise_dis)
    extra["episode_extra_stats/camera_noise_disabled_static"] = float(static_noise_dis)
    extra["episode_extra_stats/camera_frame_dropout_disabled_drone"] = float(drone_fd_dis)
    extra["episode_extra_stats/camera_frame_dropout_disabled_static"] = float(static_fd_dis)
    task.infos["episode_extra_stats"] = extra
