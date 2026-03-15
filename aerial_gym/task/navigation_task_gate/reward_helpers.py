from __future__ import annotations

import os
import torch

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import *  # noqa: F401,F403

logger = CustomLogger("navigation_task_gate_rewards")


def _detect_boundary_violation(task, robot_position: torch.Tensor) -> torch.Tensor:
    """Detect one-shot boundary violations (crossing gate plane outside passage window)."""
    y_margin = 0.2
    behind_gate_mask = robot_position[:, 1] > (task.gate_position[:, 1] + y_margin)
    gate_passage_width_tolerance = task.gate_width * 0.5
    gate_min_height = task.gate_position[:, 2] + task.gate_height * 0.0
    gate_max_height = task.gate_position[:, 2] + task.gate_height * 1.0
    within_passage_window = (
        (torch.abs(robot_position[:, 0] - task.gate_position[:, 0]) < gate_passage_width_tolerance)
        & (robot_position[:, 2] > gate_min_height)
        & (robot_position[:, 2] < gate_max_height)
    )
    misaligned_cross_mask = behind_gate_mask & (~within_passage_window) & (~task.gate_passed)
    if not True:
        task._bv_flag_episode = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
    boundary_violation_one_shot_mask = misaligned_cross_mask & (~task._bv_flag_episode)
    task._bv_flag_episode |= boundary_violation_one_shot_mask
    return boundary_violation_one_shot_mask

def _apply_time_penalty(task, rewards: torch.Tensor, robot_position: torch.Tensor) -> torch.Tensor:
    """Compute and apply per-step time penalty scaled by curriculum."""
    # Per-step time cost (scaled like other dense shaping)
    # r_time = -lambda0 * (1 + lambda1 * s^p),  s = step / horizon
    rp = task.task_config.reward_parameters
    try:
        H = int(task.task_config.episode_len_steps)
        if H <= 0:
            H = 100
    except (ValueError, TypeError):
        H = 100
    lam1 = float(rp.get('time_penalty_lambda1', 1.0))
    p = float(rp.get('time_penalty_exponent', 2.0))
    desired_total = float(rp.get('time_penalty_total_at_horizon', 4.0))
    lam0 = rp.get('time_penalty_lambda0', None)
    if lam0 is None:
        if abs(p - 2.0) < 1e-6:
            sum_s_p = ((H + 1.0) * (2.0 * H + 1.0)) / (6.0 * H)
        else:
            sum_s_p = H / (p + 1.0)
        denom = H + lam1 * sum_s_p
        lam0 = desired_total / max(denom, 1e-6)
    lam0 = float(lam0)
    try:
        s = torch.clamp((task.episode_lengths + 1.0) / float(H), 0.0, 1.0)
    except (ValueError, TypeError):
        s = torch.full((task.num_envs,), 1.0 / float(H), device=task.device)
    raw_time_penalty = -lam0 * (1.0 + lam1 * torch.pow(s, p))
    reward_scale = float(rp.get('reward_scale', 0.1))
    try:
        mult_factor = float(task._curriculum_multiplier_factor)
    except (ValueError, TypeError):
        mult_factor = 1.0
    time_penalty = (mult_factor * reward_scale * raw_time_penalty)
    rewards = rewards + time_penalty
    # Accumulate for per-episode debugging
    non_terminated = ~task.terminations
    if torch.sum(non_terminated) > 0:
        task.episode_time_penalty[non_terminated] += time_penalty[non_terminated]

    return rewards

def _apply_static_fov_reward(task, rewards: torch.Tensor, robot_position: torch.Tensor) -> torch.Tensor:
    """Apply static camera FOV visibility reward if enabled."""
    # Static camera FOV visibility reward (depth-based frustum check, shaped)
    try:
        try:
            fov_mag = float(task.task_config.reward_parameters.get("static_fov_visibility_reward_magnitude", 0.0))
        except (ValueError, TypeError):
            fov_mag = 0.0
        # If env var SF_ENABLE_STATIC_FOV_REWARD is not explicitly true, force-disable
        try:
            _env_flag = os.environ.get('SF_ENABLE_STATIC_FOV_REWARD', '').strip().lower()
            if _env_flag not in ('1', 'true', 'yes', 'y'):
                fov_mag = 0.0
        except (KeyError, TypeError):
            fov_mag = 0.0
        try:
            fov_alpha = float(task.task_config.reward_parameters.get("static_fov_visibility_exponent", 2.0))
        except (ValueError, TypeError):
            fov_alpha = 2.0
        if fov_mag != 0.0:
            # Camera base position (x=0, y=base_y, z either adaptive gate center or fixed 1.5)
            try:
                parent = task.sim_env
                gtd = parent.global_tensor_dict if (parent is not None and hasattr(parent, 'global_tensor_dict')) else {}
            except (AttributeError, KeyError):
                gtd = {}
            try:
                base_y = float(os.environ.get('SF_STATIC_CAMERA_BASE_Y', gtd.get('static_camera/base_y', -3.0)))
            except (ValueError, TypeError):
                base_y = -3.0
            try:
                base_z_env = os.environ.get('SF_STATIC_CAMERA_BASE_Z', None)
                if base_z_env is None:
                    base_z_env = gtd.get('static_camera/base_z', 1.5)
                adaptive_z = isinstance(base_z_env, str) and base_z_env.strip().lower() == 'adaptive'
            except (KeyError, TypeError):
                adaptive_z = False

            # Resolve Z per env
            if adaptive_z:
                gate_center_z = task.gate_center_height
            else:
                gate_center_z = torch.full((task.num_envs,), 1.5, device=task.device)

            cam_pos = torch.stack([
                torch.zeros(task.num_envs, device=task.device),
                torch.full((task.num_envs,), base_y, device=task.device),
                gate_center_z
            ], dim=1)
            target = torch.stack([
                torch.zeros(task.num_envs, device=task.device),
                torch.zeros(task.num_envs, device=task.device),
                gate_center_z
            ], dim=1)

            # Camera basis (right, up, forward)
            fwd = target - cam_pos
            fwd = fwd / (torch.norm(fwd, dim=1, keepdim=True) + 1e-8)
            up_world = torch.tensor([0.0, 0.0, 1.0], device=task.device).view(1, 3).expand_as(fwd)
            right = torch.cross(fwd, up_world)
            right = right / (torch.norm(right, dim=1, keepdim=True) + 1e-8)
            up = torch.cross(right, fwd)

            # Transform drone position into camera coordinates
            pw = robot_position - cam_pos
            x_c = torch.sum(pw * right, dim=1)
            y_c = torch.sum(pw * up, dim=1)
            z_c = torch.sum(pw * fwd, dim=1)

            # Visibility checks within symmetric FOV (approx for D455)
            half_fov_rad = (87.0 * 3.141592653589793 / 180.0) * 0.5
            horiz_angle = torch.atan2(torch.abs(x_c), torch.clamp(z_c, min=1e-6))
            vert_angle = torch.atan2(torch.abs(y_c), torch.clamp(z_c, min=1e-6))
            visible = (z_c > 0.1) & (horiz_angle <= half_fov_rad) & (vert_angle <= half_fov_rad)

            # Graded score inside frustum: m = max(h/hfov, v/vfov); score = (1 - m)^alpha
            h_norm = torch.clamp(horiz_angle / half_fov_rad, 0.0, 1.0)
            v_norm = torch.clamp(vert_angle / half_fov_rad, 0.0, 1.0)
            m_norm = torch.maximum(h_norm, v_norm)
            fov_score = torch.pow(torch.clamp(1.0 - m_norm, min=0.0), fov_alpha)
            fov_reward = fov_mag * fov_score

            # Apply only for non-terminated envs and visible
            add_mask = visible & (~task.terminations)
            if torch.any(add_mask):
                rewards[add_mask] = rewards[add_mask] + fov_reward[add_mask]
                # Track episode totals
                task.episode_static_fov_visibility_reward[add_mask] += fov_reward[add_mask]

            # Periodic debug infos
            if(task.num_task_steps % 200 == 0):
                frac_visible = float(torch.mean(visible.float()).item())
                avg_h = float(torch.mean(horiz_angle).item())
                avg_v = float(torch.mean(vert_angle).item())
                avg_score = float(torch.mean(fov_score).item())
                task.infos["static_fov/visible_fraction"] = torch.tensor(frac_visible, dtype=torch.float32)
                task.infos["static_fov/avg_horiz_angle_rad"] = torch.tensor(avg_h, dtype=torch.float32)
                task.infos["static_fov/avg_vert_angle_rad"] = torch.tensor(avg_v, dtype=torch.float32)
                task.infos["static_fov/avg_score"] = torch.tensor(avg_score, dtype=torch.float32)

                # Per-step env0 deep dive
                try:
                    env0 = 0
                    if visible.shape[0] > env0:
                        vis0 = bool(visible[env0].item())
                        hdeg0 = float(horiz_angle[env0].item() * (180.0 / 3.141592653589793))
                        vdeg0 = float(vert_angle[env0].item() * (180.0 / 3.141592653589793))
                        xc0 = float(x_c[env0].item()); yc0 = float(y_c[env0].item()); zc0 = float(z_c[env0].item())
                        score0 = float(fov_score[env0].item())
                        # Current static camera yaw (if available)
                        try:
                            scm = task.static_camera_manager
                            yaw_cur = float(scm.current_camera_angles[env0]) if (scm is not None and hasattr(scm, 'current_camera_angles') and len(scm.current_camera_angles) > env0) else 0.0
                        except (ValueError, TypeError):
                            yaw_cur = 0.0
                        logger.warning(f" 🖼️ env0 FOV: visible={1 if vis0 else 0} h={hdeg0:.1f}° v={vdeg0:.1f}° | score={score0:.3f} | cam_yaw={yaw_cur:.1f}° | x_c={xc0:.2f}, y_c={yc0:.2f}, z_c={zc0:.2f}")
                except (ValueError, TypeError):
                    pass
    except (ValueError, TypeError):
        pass
    return rewards

def post_image_reward_addition(task) -> None:
    """Add image-based rewards from drone camera."""
    image_obs = task.obs_dict["depth_range_pixels"].squeeze(1)
    image_obs[image_obs < 0] = 10.0
    task.min_pixel_dist = torch.amin(image_obs, dim=(1, 2))
    
    # Calculate image rewards for debugging
    mag = task.task_config.reward_parameters.get("image_penalty_magnitude", 4.0)
    expo = task.task_config.reward_parameters.get("image_penalty_exponent", 1.0)
    image_rewards = -exponential_reward_function(
        float(mag), float(expo), task.min_pixel_dist[~task.terminations]
    )
    # Apply image penalty to non-terminated environments
    task.rewards[~task.terminations] += image_rewards
    
    # COMPREHENSIVE IMAGE REWARD DEBUGGING: Print values every 200 steps  
    if task.num_task_steps % 200 == 0:
        avg_min_dist = torch.mean(task.min_pixel_dist).item()
        avg_image_reward = torch.mean(image_rewards).item() if len(image_rewards) > 0 else 0.0
        min_pixel_dist = torch.min(task.min_pixel_dist).item()
        max_pixel_dist = torch.max(task.min_pixel_dist).item()
        
        # Count environments with different distance ranges
        very_close_count = torch.sum(task.min_pixel_dist < 2.0).item()  # < 2m
        close_count = torch.sum((task.min_pixel_dist >= 2.0) & (task.min_pixel_dist < 4.0)).item()  # 2-4m
        safe_count = torch.sum(task.min_pixel_dist >= 4.0).item()  # > 4m
        
    # Apply the image rewards
    task.rewards[~task.terminations] += image_rewards

