from __future__ import annotations

import torch

from aerial_gym.task.navigation_task_gate.reward_functions import compute_gate_reward
from aerial_gym.task.schemas import GATE_OBS_LAYOUT
from aerial_gym.utils.env_flag_utils import read_env_bool
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import get_euler_xyz_tensor, quat_rotate_inverse, ssa

logger = CustomLogger("navigation_task_gate_obs_reward")


def logging_sanity_check(
    task: object,
    infos: dict[str, torch.Tensor],
) -> None:
    """Sanity check for logging to detect issues with success/crash/timeout logic."""
    successes = infos["successes"]
    crashes = infos["crashes"]
    timeouts = infos["timeouts"]
    time_at_crash = torch.where(
        crashes > 0,
        task.sim_env.sim_steps,
        task.task_config.episode_len_steps * torch.ones_like(task.sim_env.sim_steps),
    )
    env_list_for_toc = (time_at_crash < 5).nonzero(as_tuple=False).squeeze(-1)
    crash_envs = crashes.nonzero(as_tuple=False).squeeze(-1)
    success_envs = successes.nonzero(as_tuple=False).squeeze(-1)
    timeout_envs = timeouts.nonzero(as_tuple=False).squeeze(-1)  # noqa: F841

    if len(env_list_for_toc) > 0:
        logger.critical("Crash is happening too soon.")
        logger.critical(f"Envs crashing too soon: {env_list_for_toc}")
        logger.critical(f"Time at crash: {time_at_crash[env_list_for_toc]}")

    if torch.sum(torch.logical_and(successes, crashes)) > 0:
        logger.critical("Success and crash are occuring at the same time")
        logger.critical(
            f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
        )
        logger.critical(
            f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
        )
        logger.critical(
            f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, successes))}"
        )


def _apply_state_noise(
    task: object,
    tensor: torch.Tensor,
    noise_key: str,
    wrap_angles: bool = False,
) -> torch.Tensor:
    """Apply curriculum-driven state noise to a tensor if enabled."""
    if not task.task_config.curriculum.enable_state_noise:
        return tensor
    if bool(task.sim_env.global_tensor_dict.get("state_randomization/noise_disabled", False)):
        return tensor
    noise_cfg = task.task_config.curriculum.get_state_noise(task.curriculum_level)
    std = float(noise_cfg.get(noise_key, 0.0))
    if std <= 0.0:
        return tensor
    result = tensor + torch.randn_like(tensor) * std
    if wrap_angles:
        result = torch.atan2(torch.sin(result), torch.cos(result))
    return result


def process_obs_for_task(task: object) -> None:
    """Assemble 150D observation vector from raw sensor/state data (see GATE_OBS_LAYOUT)."""
    drone_pos_clean = task.obs_dict["robot_position"]
    drone_pos_noised = _apply_state_noise(task, drone_pos_clean, "drone_pos_std_m")

    obs = task.task_obs["observations"]
    layout = GATE_OBS_LAYOUT

    obs[:, layout.drone_position] = drone_pos_noised

    static_camera_pos, static_camera_orientation = (
        task._camera._get_static_camera_pose_relative_to_drone()
    )
    static_camera_pos = _apply_state_noise(task, static_camera_pos, "static_pos_std_m")
    static_camera_orientation = _apply_state_noise(
        task, static_camera_orientation, "static_orient_std_rad", wrap_angles=True
    )

    obs[:, layout.static_camera_position] = static_camera_pos
    obs[:, layout.static_camera_orientation] = static_camera_orientation

    euler_angles = ssa(get_euler_xyz_tensor(task.obs_dict["robot_vehicle_orientation"]))
    euler_angles = _apply_state_noise(task, euler_angles, "drone_orient_std_rad", wrap_angles=True)
    obs[:, layout.drone_orientation] = euler_angles

    obs[:, layout.body_linear_velocity] = task.obs_dict["robot_body_linvel"]
    obs[:, layout.body_angular_velocity] = task.obs_dict["robot_body_angvel"]
    obs[:, layout.actions] = task.obs_dict["robot_actions"]

    if isinstance(task.image_latents, torch.Tensor) and task.image_latents.shape[1] >= 64:
        obs[:, layout.drone_vae_latents] = task.image_latents[:, :64]
    if (
        isinstance(task.static_image_latents, torch.Tensor)
        and task.static_image_latents.shape[1] >= 64
    ):
        obs[:, layout.static_vae_latents] = task.static_image_latents[:, :64]

    obs_tensor = task.task_obs.get("observations", None)
    if isinstance(obs_tensor, torch.Tensor):
        bad = ~torch.isfinite(obs_tensor)
        if torch.any(bad):
            if task.task_config.guard_debug_enabled:
                logger.warning(
                    f"[NaNGuard] Sanitizing {int(torch.sum(bad).item())} invalid obs entries before return."
                )
            obs_tensor[bad] = 0.0


def update_camera_modes(task: object) -> None:
    """Update camera modes in priority order: arc-follow > dynamic-follow > yaw-sweep/locked-follow."""
    dynamic_enabled = task.task_config.curriculum.enable_dynamic_camera_following
    dynamic_disabled = bool(
        task.sim_env.global_tensor_dict.get("dynamic_camera_following/disabled", False)
    )
    arc_follow_enabled = bool(
        task.sim_env.global_tensor_dict.get("static_camera/arc_follow_enabled", False)
    )

    if arc_follow_enabled:
        task.static_camera_manager.update_arc_follow(
            task.obs_dict["robot_position"],
            task.gate_position,
            task.gate_center_height,
            float(task.sim_env.global_tensor_dict.get("static_camera/arc_follow_radius_m", 2.0)),
        )
    elif dynamic_enabled and not dynamic_disabled:
        task.static_camera_manager.update_dynamic_camera_following(
            task.obs_dict["robot_position"], task.gate_position, task.gate_center_height
        )

    gtd = task.sim_env.global_tensor_dict
    sweep_enabled_flag = str(gtd.get("static_camera/yaw_sweep_enabled", "false")).lower() == "true"
    locked_follow = bool(gtd.get("static_camera/locked_follow", False))
    static_mode_active = not (dynamic_enabled and not dynamic_disabled) and not arc_follow_enabled
    if locked_follow and static_mode_active:
        task.static_camera_manager.update_locked_follow(task.obs_dict["robot_position"])
    elif sweep_enabled_flag and static_mode_active:
        env_ids_all = torch.arange(task.sim_env.num_envs, device=task.device)
        task.static_camera_manager.update_camera_positions(task.curriculum_level, env_ids_all)


def compute_rewards_and_crashes(
    task: object,
    obs_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute rewards with gate-specific components."""
    robot_position = obs_dict["robot_position"]
    target_position = task.target_position
    robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]

    task.pos_error_vehicle_frame_prev[:] = task.pos_error_vehicle_frame
    task.pos_error_vehicle_frame[:] = quat_rotate_inverse(
        robot_vehicle_orientation, (target_position - robot_position)
    )

    current_actions = obs_dict["robot_actions"].clone()
    previous_actions = obs_dict["robot_prev_actions"].clone()

    prev_actions_for_reward = previous_actions
    fresh_mask = task._episode_fresh
    if isinstance(fresh_mask, torch.Tensor) and fresh_mask.shape[0] == task.num_envs:
        if torch.any(fresh_mask):
            task.pos_error_vehicle_frame_prev[fresh_mask] = task.pos_error_vehicle_frame[fresh_mask]
            prev_actions_for_reward = previous_actions.clone()
            prev_actions_for_reward[fresh_mask] = current_actions[fresh_mask]

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
    task._curriculum_multiplier_factor = 1.0 + 0.5 * frac_eff

    boundary_violation_one_shot_mask = task._rewards._detect_boundary_violation(robot_position)

    with torch.jit.optimized_execution(False):
        rewards, crashes, camera_gate_alignment = compute_gate_reward(
            task.pos_error_vehicle_frame,
            task.pos_error_vehicle_frame_prev,
            obs_dict["crashes"],
            current_actions,
            prev_actions_for_reward,
            robot_position,
            robot_vehicle_orientation,
            task.gate_position,
            task.gate_passed,
            frac_eff,
            task.task_config.reward_parameters,
            task.gate_width,
            task.gate_height,
            task.gate_center_height,
            boundary_violation_one_shot_mask,
        )

    rewards = task._rewards._apply_time_penalty(rewards, robot_position)
    rewards = task._rewards._apply_static_fov_reward(rewards, robot_position)
    task._reward_tracking.update_episode_reward_tracking(obs_dict, rewards, crashes)
    task._reward_tracking._log_comprehensive_reward_debug(
        obs_dict, rewards, crashes, boundary_violation_one_shot_mask, camera_gate_alignment
    )

    task.camera_alignment_debug = camera_gate_alignment

    return rewards, crashes, camera_gate_alignment
