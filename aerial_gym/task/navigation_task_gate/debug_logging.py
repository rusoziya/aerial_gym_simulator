from __future__ import annotations

import os
import torch

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("navigation_task_gate_debug")


def _log_comprehensive_reward_debug(
    self,
    obs_dict: dict[str, torch.Tensor],
    rewards: torch.Tensor,
    crashes: torch.Tensor,
    boundary_violation_one_shot_mask: torch.Tensor,
    camera_gate_alignment: torch.Tensor,
) -> None:
    """Recalculate and log all reward components (every 200 steps, gated by config flag)."""
    # COMPREHENSIVE REWARD DEBUGGING: Print ALL reward components every 200 steps
    # Disabled by default via config flag `enable_comprehensive_reward_debug`
    if (
        True
        and task.num_task_steps % 200 == 0
        and bool(task.task_config.enable_comprehensive_reward_debug)
    ):
        # Recalculate components for debugging (without JIT optimization)
        dist = torch.norm(task.pos_error_vehicle_frame, dim=1)
        prev_dist = torch.norm(task.pos_error_vehicle_frame_prev, dim=1)
        action = obs_dict["robot_actions"]
        prev_action = obs_dict["robot_prev_actions"]
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        
        # Individual reward components (average across environments)
        pos_reward = exponential_reward_function(
            task.task_config.reward_parameters["pos_reward_magnitude"],
            task.task_config.reward_parameters["pos_reward_exponent"],
            dist,
        )
        
        very_close_reward = exponential_reward_function(
            task.task_config.reward_parameters["very_close_to_goal_reward_magnitude"],
            task.task_config.reward_parameters["very_close_to_goal_reward_exponent"],
            dist,
        )
        
        getting_closer = prev_dist - dist
        getting_closer_reward = torch.where(
            getting_closer > 0,
            task.task_config.reward_parameters["getting_closer_reward_multiplier"] * getting_closer,
            2.0 * task.task_config.reward_parameters["getting_closer_reward_multiplier"] * getting_closer,
        )
        
        # Use adaptive gate center (z = bottom + center_height)
        gate_center_position = task.gate_position.clone()
        gate_center_position[:, 2] = gate_center_position[:, 2] + task.gate_center_height
        gate_distance = torch.norm(robot_position - gate_center_position, dim=1)
        gate_approach_reward = exponential_reward_function(
            task.task_config.reward_parameters["gate_approach_reward_magnitude"],
            0.5,
            gate_distance,
        )
        
        # Gate alignment (finer piecewise by lateral offset from gate center)
        gate_alignment_reward = torch.zeros_like(gate_distance)
        dx = torch.abs(robot_position[:, 0] - task.gate_position[:, 0])
        # Thresholds proportional to current gate width (denser piecewise bins)
        t00 = task.gate_width * 0.01
        t0  = task.gate_width * 0.02
        t1  = task.gate_width * 0.04
        t2  = task.gate_width * 0.06
        t3  = task.gate_width * 0.08
        t4  = task.gate_width * 0.10
        t5  = task.gate_width * 0.12
        t6  = task.gate_width * 0.15
        t7  = task.gate_width * 0.20
        t8  = task.gate_width * 0.25
        t9  = task.gate_width * 0.30
        t10 = task.gate_width * 0.40
        t11 = task.gate_width * 0.50
        mag = task.task_config.reward_parameters["gate_alignment_reward_magnitude"]
        gate_alignment_reward[dx <= t00] = 1.00 * mag
        gate_alignment_reward[(dx > t00) & (dx <= t0)] = 0.97 * mag
        gate_alignment_reward[(dx > t0) & (dx <= t1)]  = 0.94 * mag
        gate_alignment_reward[(dx > t1) & (dx <= t2)]  = 0.90 * mag
        gate_alignment_reward[(dx > t2) & (dx <= t3)]  = 0.85 * mag
        gate_alignment_reward[(dx > t3) & (dx <= t4)]  = 0.80 * mag
        gate_alignment_reward[(dx > t4) & (dx <= t5)]  = 0.72 * mag
        gate_alignment_reward[(dx > t5) & (dx <= t6)]  = 0.65 * mag
        gate_alignment_reward[(dx > t6) & (dx <= t7)]  = 0.55 * mag
        gate_alignment_reward[(dx > t7) & (dx <= t8)]  = 0.45 * mag
        gate_alignment_reward[(dx > t8) & (dx <= t9)]  = 0.35 * mag
        gate_alignment_reward[(dx > t9) & (dx <= t10)] = 0.25 * mag
        gate_alignment_reward[(dx > t10) & (dx <= t11)] = 0.15 * mag
        
        # Camera facing reward calculation (same as in compute_gate_reward)
        drone_to_gate = task.gate_position - robot_position
        drone_to_gate_normalized = drone_to_gate / (torch.norm(drone_to_gate, dim=1, keepdim=True) + 1e-8)
        
        # Get drone's forward direction (where camera points)
        qw, qx, qy, qz = robot_vehicle_orientation[:, 3], robot_vehicle_orientation[:, 0], robot_vehicle_orientation[:, 1], robot_vehicle_orientation[:, 2]
        forward_x = 1.0 - 2.0 * (qy * qy + qz * qz)
        forward_y = 2.0 * (qx * qy + qw * qz)
        forward_z = 2.0 * (qx * qz - qw * qy)
        drone_forward = torch.stack([forward_x, forward_y, forward_z], dim=1)
        drone_forward_normalized = drone_forward / (torch.norm(drone_forward, dim=1, keepdim=True) + 1e-8)
        
        # Calculate alignment between camera direction and gate direction
        camera_gate_alignment = torch.sum(drone_forward_normalized * drone_to_gate_normalized, dim=1)
        camera_gate_alignment = torch.clamp(camera_gate_alignment, -1.0, 1.0)
        
        # Camera facing reward with same logic as compute_gate_reward
        camera_facing_reward = torch.zeros_like(camera_gate_alignment)
        perfect_mask = camera_gate_alignment > 0.966
        camera_facing_reward[perfect_mask] = task.task_config.reward_parameters["camera_facing_reward_magnitude"]
        excellent_mask = (camera_gate_alignment > 0.866) & (camera_gate_alignment <= 0.966)
        camera_facing_reward[excellent_mask] = 0.9 * task.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[excellent_mask]
        good_mask = (camera_gate_alignment > 0.5) & (camera_gate_alignment <= 0.866)
        camera_facing_reward[good_mask] = 0.8 * task.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[good_mask]
        moderate_mask = (camera_gate_alignment > 0.0) & (camera_gate_alignment <= 0.5)
        camera_facing_reward[moderate_mask] = 0.4 * task.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[moderate_mask]
        poor_mask = (camera_gate_alignment > -0.707) & (camera_gate_alignment <= 0.0)
        camera_facing_reward[poor_mask] = 0.2 * task.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[poor_mask]
        severe_mask = camera_gate_alignment <= -0.707
        camera_facing_reward[severe_mask] = 2.0 * task.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[severe_mask]
        
        # Action penalties - FIXED: Added missing Y-action penalties for 4D action space
        action_diff = action - prev_action
        
        # ENHANCED ACTION DEBUG: Deep investigation of action tracking system
        if task.num_task_steps % 200 == 0:
            avg_action_diff = torch.mean(torch.abs(action_diff), dim=0)
            max_action_diff = torch.max(torch.abs(action_diff), dim=0)[0]
            
            # Show actual action values to understand the pattern
            avg_current = torch.mean(action, dim=0)
            avg_previous = torch.mean(prev_action, dim=0)
            
            # Check if all actions are identical across environments
            action_std = torch.std(action, dim=0)
            prev_action_std = torch.std(prev_action, dim=0)
            
        x_diff_penalty = exponential_penalty_function(
            task.task_config.reward_parameters["x_action_diff_penalty_magnitude"],
            task.task_config.reward_parameters["x_action_diff_penalty_exponent"],
            action_diff[:, 0],
        )
        y_diff_penalty = exponential_penalty_function(
            task.task_config.reward_parameters["y_action_diff_penalty_magnitude"],
            task.task_config.reward_parameters["y_action_diff_penalty_exponent"],
            action_diff[:, 1],
        )
        z_diff_penalty = exponential_penalty_function(
            task.task_config.reward_parameters["z_action_diff_penalty_magnitude"],
            task.task_config.reward_parameters["z_action_diff_penalty_exponent"],
            action_diff[:, 2],
        )
        yawrate_diff_penalty = exponential_penalty_function(
            task.task_config.reward_parameters["yawrate_action_diff_penalty_magnitude"],
            task.task_config.reward_parameters["yawrate_action_diff_penalty_exponent"],
            action_diff[:, 3],
        )
        
        x_absolute_penalty = exponential_penalty_function(
            task.task_config.reward_parameters["x_absolute_action_penalty_magnitude"],
            task.task_config.reward_parameters["x_absolute_action_penalty_exponent"],
            action[:, 0],
        )
        y_absolute_penalty = exponential_penalty_function(
            task.task_config.reward_parameters["y_absolute_action_penalty_magnitude"],
            task.task_config.reward_parameters["y_absolute_action_penalty_exponent"],
            action[:, 1],
        )
        z_absolute_penalty = exponential_penalty_function(
            task.task_config.reward_parameters["z_absolute_action_penalty_magnitude"],
            task.task_config.reward_parameters["z_absolute_action_penalty_exponent"],
            action[:, 2],
        )
        yawrate_absolute_penalty = exponential_penalty_function(
            task.task_config.reward_parameters["yawrate_absolute_action_penalty_magnitude"],
            task.task_config.reward_parameters["yawrate_absolute_action_penalty_exponent"],
            action[:, 3],
        )
        
        action_diff_penalty = x_diff_penalty + y_diff_penalty + z_diff_penalty + yawrate_diff_penalty
        absolute_action_penalty = x_absolute_penalty + y_absolute_penalty + z_absolute_penalty + yawrate_absolute_penalty
        total_action_penalty = action_diff_penalty + absolute_action_penalty
        
        # Calculate averages for debugging
        mult_factor = 1.0 + (0.5) * task.curriculum_progress_fraction
        avg_total_reward = torch.mean(rewards).item()
        # Use the effective multiplier factor computed earlier in this step
        try:
            mult_factor = float(task._curriculum_multiplier_factor)
        except (ValueError, TypeError):
            mult_factor = 1.0
        avg_pos_reward = torch.mean(mult_factor * pos_reward).item()
        avg_very_close = torch.mean(mult_factor * very_close_reward).item()
        avg_getting_closer = torch.mean(mult_factor * getting_closer_reward).item()
        avg_gate_approach = torch.mean(mult_factor * gate_approach_reward).item()
        avg_gate_alignment = torch.mean(mult_factor * gate_alignment_reward).item()
        avg_camera_facing = torch.mean(mult_factor * camera_facing_reward).item()
        avg_action_penalty = torch.mean(total_action_penalty).item()
        # Boundary violation penalty: calculate for debugging (same logic as torchscript)
        y_margin = 0.2
        behind_gate_mask = (robot_position[:, 1] > (task.gate_position[:, 1] + y_margin))
        gate_passage_width_tolerance = task.gate_width * 0.6
        gate_min_height = task.gate_position[:, 2] + task.gate_height * 0.1
        gate_max_height = task.gate_position[:, 2] + task.gate_height * 0.9
        within_passage_window = (
            (torch.abs(robot_position[:, 0] - task.gate_position[:, 0]) < gate_passage_width_tolerance)
            & (robot_position[:, 2] > gate_min_height)
            & (robot_position[:, 2] < gate_max_height)
        )
        misaligned_cross_mask = behind_gate_mask & (~within_passage_window) & (~task.gate_passed)
        boundary_violation_penalty = torch.zeros_like(gate_distance)
        boundary_violation_penalty[misaligned_cross_mask] = -50.0
        avg_boundary_penalty = torch.mean(boundary_violation_penalty).item()
        avg_distance = torch.mean(dist).item()
        avg_gate_distance = torch.mean(gate_distance).item()
        avg_camera_alignment = torch.mean(camera_gate_alignment).item()
        # Static FOV (recompute shaped-average for logging)
        try:
            fov_mag = float(task.task_config.reward_parameters.get("static_fov_visibility_reward_magnitude", 0.0))
        except (ValueError, TypeError):
            fov_mag = 0.0
        avg_static_fov_reward = 0.0
        if fov_mag != 0.0:
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
            fwd = target - cam_pos
            fwd = fwd / (torch.norm(fwd, dim=1, keepdim=True) + 1e-8)
            up_world = torch.tensor([0.0, 0.0, 1.0], device=task.device).view(1, 3).expand_as(fwd)
            right = torch.cross(fwd, up_world); right = right / (torch.norm(right, dim=1, keepdim=True) + 1e-8)
            up = torch.cross(right, fwd)
            pw = robot_position - cam_pos
            x_c = torch.sum(pw * right, dim=1); y_c = torch.sum(pw * up, dim=1); z_c = torch.sum(pw * fwd, dim=1)
            half_fov_rad = (87.0 * 3.141592653589793 / 180.0) * 0.5
            horiz_angle = torch.atan2(torch.abs(x_c), torch.clamp(z_c, min=1e-6))
            vert_angle = torch.atan2(torch.abs(y_c), torch.clamp(z_c, min=1e-6))
            visible = (z_c > 0.1) & (horiz_angle <= half_fov_rad) & (vert_angle <= half_fov_rad)
            h_norm = torch.clamp(horiz_angle / half_fov_rad, 0.0, 1.0)
            v_norm = torch.clamp(vert_angle / half_fov_rad, 0.0, 1.0)
            m_norm = torch.maximum(h_norm, v_norm)
            try:
                fov_alpha = float(task.task_config.reward_parameters.get("static_fov_visibility_exponent", 2.0))
            except (ValueError, TypeError):
                fov_alpha = 2.0
            fov_score = torch.pow(torch.clamp(1.0 - m_norm, min=0.0), fov_alpha)
            avg_static_fov_reward = float(torch.mean(fov_mag * fov_score).item())
        
        logger.warning("="*80)
        logger.warning(f"🔍 COMPREHENSIVE REWARD BREAKDOWN (Step {task.num_task_steps}):")
        logger.warning(f"  📊 TOTAL REWARD:           {avg_total_reward:.3f}")
        # Print VAE latent statistics alongside reward breakdown for clear visibility
        if isinstance(task.task_obs, dict) and 'observations' in task.task_obs:
            obs_all = task.task_obs['observations']
            if obs_all.shape[1] >= 150:
                d_lat = obs_all[:, 22:86]
                s_lat = obs_all[:, 86:150]
                d_mu = float(torch.mean(d_lat).item()); s_mu = float(torch.mean(s_lat).item())
                d_std = float(torch.std(d_lat).item()); s_std = float(torch.std(s_lat).item())
                d_norm = float(torch.linalg.norm(d_lat).item()); s_norm = float(torch.linalg.norm(s_lat).item())
                ratio = s_norm / (d_norm + 1e-6)
                logger.warning(
                    f"  🔬 VAE STATS: norm_ratio={ratio:.3f} | "
                    f"drone(mu={d_mu:.3f}, std={d_std:.3f}) | static(mu={s_mu:.3f}, std={s_std:.3f})"
                )
        logger.warning(f"  📍 Position Reward:        {avg_pos_reward:.3f} (dist: {avg_distance:.2f}m)")
        logger.warning(f"  🎯 Very Close Reward:      {avg_very_close:.3f}")
        logger.warning(f"  ⬆️  Getting Closer:         {avg_getting_closer:.3f}")
        logger.warning(f"  🚪 Gate Approach:          {avg_gate_approach:.3f} (gate_dist: {avg_gate_distance:.2f}m)")
        logger.warning(f"  ✅ Gate Alignment:         {avg_gate_alignment:.3f}")
        logger.warning(f"  📹 Camera Facing:          {avg_camera_facing:.3f} (align: {avg_camera_alignment:.3f})")
        if fov_mag != 0.0:
            logger.warning(f"  🖼️ Static FOV Reward:      {avg_static_fov_reward:.3f}")
        logger.warning(f"  🎮 Action Penalty:         {avg_action_penalty:.3f}")
        # Time/timeout penalties (averages)
        try:
            avg_time_pen = float(torch.mean(task.episode_time_penalty).item())
        except (ValueError, TypeError):
            avg_time_pen = 0.0
        try:
            avg_timeout_pen = float(torch.mean(task.episode_timeout_penalty).item())
        except (ValueError, TypeError):
            avg_timeout_pen = 0.0
        logger.warning(f"  ⏱️ Time Penalty (avg):     {avg_time_pen:.3f}")
        logger.warning(f"  ⌛ Timeout Penalty (avg):  {avg_timeout_pen:.3f}")
        logger.warning(f"  ⛔ Boundary Violation:     {avg_boundary_penalty:.3f}")
        logger.warning(f"  ⚡ Multiplier Factor:      {mult_factor:.3f}")
        
        # Check for any gate passages - ADAPTIVE to gate dimensions
        curriculum_width_tolerance = task.gate_width * 0.6  # 60% of gate width
        curriculum_min_height = task.gate_position[:, 2] + task.gate_height * 0.08  # 8% above ground
        curriculum_max_height = task.gate_position[:, 2] + task.gate_height * 0.92  # 92% of gate height
        
        num_passed = torch.sum((robot_position[:, 1] > task.gate_position[:, 1]) & 
                             (torch.abs(robot_position[:, 0] - task.gate_position[:, 0]) < curriculum_width_tolerance) &
                             (robot_position[:, 2] > curriculum_min_height) & (robot_position[:, 2] < curriculum_max_height)).item()
        
        if num_passed > 0:
            logger.warning(f"  🎉 GATE PASSAGES:          {num_passed}/16 environments!")
            logger.warning(f"  💰 Gate Passage Reward:   {task.task_config.reward_parameters['gate_passage_reward_magnitude'].item():.1f} per passage")
        
        # Check for crashes
        num_crashes = torch.sum(obs_dict["crashes"]).item()
        if num_crashes > 0:
            logger.warning(f"  💥 CRASHES:                {num_crashes}/16 environments")
            logger.warning(f"  💸 Collision Penalty:     {task.task_config.reward_parameters['collision_penalty'].item():.1f} per crash")
        
        # EPISODE-LEVEL REWARD BREAKDOWN: Show how components contribute to episode totals
        if len(task.completed_episodes) > 0:
            logger.warning("-"*80)
            logger.warning(f"📈 EPISODE REWARD ANALYSIS (Last {len(task.completed_episodes)} Episodes):")
            
            # Calculate averages across completed episodes
            avg_episode_data = {}
            for key in task.completed_episodes[0].keys():
                avg_episode_data[key] = sum(ep[key] for ep in task.completed_episodes) / len(task.completed_episodes)
            
            logger.warning(f"  🏆 EPISODE TOTAL:          {avg_episode_data['total_reward']:.1f}")
            logger.warning(f"  📍 Position Contribution:  {avg_episode_data['pos_reward']:.1f}")
            logger.warning(f"  🎯 Very Close Contribution: {avg_episode_data['very_close_reward']:.1f}")
            logger.warning(f"  ⬆️  Getting Closer:         {avg_episode_data['getting_closer_reward']:.1f}")
            logger.warning(f"  🚪 Gate Approach:          {avg_episode_data['gate_approach_reward']:.1f}")
            logger.warning(f"  ✅ Gate Alignment:         {avg_episode_data['gate_alignment_reward']:.1f}")
            logger.warning(f"  📹 Camera Facing:          {avg_episode_data['camera_facing_reward']:.1f}")
            logger.warning(f"  🎮 Action Penalties:       {avg_episode_data['action_penalty']:.1f}")
            # New: time-related penalties over last 10 episodes
            if 'time_penalty' in avg_episode_data:
                logger.warning(f"  ⏱️ Time Penalties:         {avg_episode_data['time_penalty']:.1f}")
            if 'timeout_penalty' in avg_episode_data:
                logger.warning(f"  ⌛ Timeout Penalties:      {avg_episode_data['timeout_penalty']:.1f}")
            if 'boundary_violation_penalty' in avg_episode_data:
                logger.warning(f"  ⛔ Boundary Violations:    {avg_episode_data['boundary_violation_penalty']:.1f}")
            logger.warning(f"  🎉 Gate Passage Bonuses:   {avg_episode_data['gate_passage_reward']:.1f} (basic + center)")
            
            # Calculate estimated passages per episode
            basic_passage_reward = 50.0  # From config
            center_bonus = 100.0  # From config
            max_reward_per_passage = (basic_passage_reward + center_bonus) * 1.5  # With curriculum multiplier
            estimated_passages = avg_episode_data['gate_passage_reward'] / max_reward_per_passage
            logger.warning(f"  📊 Estimated Passages:     {estimated_passages:.1f} per episode (should be ≤1.0)")
            logger.warning(f"  💥 Collision Penalties:    {avg_episode_data['collision_penalty']:.1f}")
            logger.warning(f"  📷 Image Penalties:        {avg_episode_data['image_reward']:.1f}")
            logger.warning(f"  📏 Average Episode Length: {avg_episode_data['episode_length']:.0f} steps")
            
            # Show recent trend (if we have enough episodes)
            if len(task.completed_episodes) >= 5:
                recent_total = sum(ep['total_reward'] for ep in task.completed_episodes[-3:]) / 3
                older_total = sum(ep['total_reward'] for ep in task.completed_episodes[:3]) / 3
                trend = recent_total - older_total
                trend_emoji = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                logger.warning(f"  {trend_emoji} Recent Trend:         {trend:+.1f} (last 3 vs first 3)")
        
        # CURRENT EPISODE PROGRESS: Show cumulative rewards for ongoing episodes
        logger.warning("-"*80)
        logger.warning("🔄 CURRENT EPISODE PROGRESS (Cumulative):")
        
        # Average current episode progress across all environments
        avg_current_pos = torch.mean(task.episode_pos_reward).item()
        avg_current_very_close = torch.mean(task.episode_very_close_reward).item()
        avg_current_getting_closer = torch.mean(task.episode_getting_closer_reward).item()
        avg_current_gate_approach = torch.mean(task.episode_gate_approach_reward).item()
        avg_current_gate_alignment = torch.mean(task.episode_gate_alignment_reward).item()
        avg_current_camera_facing = torch.mean(task.episode_camera_facing_reward).item()
        avg_current_action_penalty = torch.mean(task.episode_action_penalty).item()
        try:
            avg_current_boundary_penalty = torch.mean(task.episode_boundary_violation_penalty).item()
        except RuntimeError:
            avg_current_boundary_penalty = 0.0
        avg_current_collision_penalty = torch.mean(task.episode_collision_penalty).item()
        avg_current_episode_length = torch.mean(task.episode_lengths).item()
        
        current_total = (avg_current_pos + avg_current_very_close + avg_current_getting_closer + 
                       avg_current_gate_approach + avg_current_gate_alignment + avg_current_camera_facing + 
                       avg_current_action_penalty + avg_current_collision_penalty + avg_current_boundary_penalty)
        
        logger.warning(f"  🔄 Current Episode Total:  {current_total:.1f} (avg across 16 envs)")
        logger.warning(f"  📍 Position So Far:        {avg_current_pos:.1f}")
        logger.warning(f"  ⬆️  Getting Closer So Far:  {avg_current_getting_closer:.1f}")
        logger.warning(f"  🚪 Gate Approach So Far:   {avg_current_gate_approach:.1f}")
        logger.warning(f"  ✅ Gate Alignment So Far:  {avg_current_gate_alignment:.1f}")
        logger.warning(f"  📹 Camera Facing So Far:   {avg_current_camera_facing:.1f}")
        logger.warning(f"  💥 Collision Penalties:    {avg_current_collision_penalty:.1f}")
        logger.warning(f"  📏 Steps So Far:           {avg_current_episode_length:.0f}")
        
        logger.warning("="*80)
    
    # Reward outlier logging to catch negative spikes
    try:
        thr = float(task.task_config.reward_outlier_threshold)
    except (ValueError, TypeError):
        thr = -180.0
    try:
        if torch.any(rewards < thr):
            _bad = torch.nonzero(rewards < thr, as_tuple=False).squeeze(-1)
            if _bad.numel() > 0:
                _limit = int(task.task_config.reward_outlier_log_limit_per_step)
                _s = _bad[:_limit]
                logger.warning(f"[RewardOutlier] envs={_s.tolist()} rewards={rewards[_s].tolist()} crashes={crashes[_s].tolist()}")
                _dist = torch.norm(task.pos_error_vehicle_frame[_s], dim=1)
                _y = obs_dict['robot_position'][_s, 1]
                # Also log boundary violation mask to see if it caused spikes
                try:
                    _bv = boundary_violation_one_shot_mask[_s].tolist()
                except RuntimeError:
                    _bv = []
                logger.warning(f"[RewardOutlier] dist={_dist.tolist()} y={_y.tolist()} boundary_violation={_bv}")
    except (ValueError, TypeError):
        pass

def _log_curriculum_details(
    self,
    success_rate: float,
    crash_rate: float,
    timeout_rate: float,
    obstacles_behind_gate: int,
    total_obstacles_in_env: int,
) -> None:
    """Log comprehensive curriculum state after level update."""
    task.log_curriculum_update(f"Gate Navigation Curriculum Level: {task.curriculum_level}, Progress: {task.curriculum_progress_fraction:.3f}")
    task.log_curriculum_update(f"\nSuccess Rate: {success_rate:.3f}\nCrash Rate: {crash_rate:.3f}\nTimeout Rate: {timeout_rate:.3f}")
    
    task.log_curriculum_update(f"\nCURRICULUM APPLIED:")
    # Report yaw sweep status in curriculum update debug (takes precedence over orientation randomization)
    try:
        yaw_enabled = str(os.environ.get('SF_ENABLE_STATIC_CAMERA_YAW_SWEEP', 'false')).lower() == 'true'
        yaw_speed = float(os.environ.get('SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG', '10.0'))
    except (ValueError, TypeError):
        yaw_enabled = False
        yaw_speed = 10.0
    # Determine dynamic camera effective state (needed below)
    try:
        dyn_cfg = task.task_config.curriculum.enable_dynamic_camera_following
        dyn_dis = bool(task.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False))
        dynamic_effective = bool(dyn_cfg and not dyn_dis)
    except (KeyError, TypeError):
        dynamic_effective = False
    # Report sweep with effective status and orientation/dynamic interactions
    if yaw_enabled and not dynamic_effective:
        task.log_curriculum_update(f"   3. STATIC CAMERA YAW SWEEP: ENABLED (speed={yaw_speed:.1f} deg/s)")
    elif yaw_enabled and dynamic_effective:
        task.log_curriculum_update(f"   3. STATIC CAMERA YAW SWEEP: ENABLED but IGNORED (dynamic camera active)")
    else:
        task.log_curriculum_update(f"   3. STATIC CAMERA YAW SWEEP: DISABLED")
    # Report arc-follow status
    try:
        arc_follow_enabled = bool(task.sim_env.global_tensor_dict.get('static_camera/arc_follow_enabled', False))
        arc_radius = float(task.sim_env.global_tensor_dict.get('static_camera/arc_follow_radius_m', 2.0))
    except (ValueError, TypeError):
        arc_follow_enabled = False
        arc_radius = 2.0
    if arc_follow_enabled:
        task.log_curriculum_update(f"   3b. STATIC CAMERA ARC-FOLLOW: ENABLED (radius={arc_radius:.1f} m)")
    if obs_dis:
        task.log_curriculum_update(f"   1. OBSTACLES: fixed to {obstacles_behind_gate} behind gate (total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)")
    else:
        task.log_curriculum_update(f"   1. OBSTACLES: {obstacles_behind_gate} behind gate (total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)")
    try:
        baseline_level = int(task.task_config.curriculum.min_level)
        pos_dis = bool(task.sim_env.global_tensor_dict.get('spawn_randomization/position_disabled', False))
        yaw_dis = bool(task.sim_env.global_tensor_dict.get('spawn_randomization/orientation_disabled', False))
        sr_active = task.task_config.curriculum.get_spawn_ranges(task.curriculum_level)
        sr_base = task.task_config.curriculum.get_spawn_ranges(baseline_level)
        sr_use = {
            'x_half_span_m': sr_base['x_half_span_m'] if pos_dis else sr_active['x_half_span_m'],
            'y_center_m':    sr_base['y_center_m']    if pos_dis else sr_active['y_center_m'],
            'y_half_span_m': sr_base['y_half_span_m'] if pos_dis else sr_active['y_half_span_m'],
            'z_center_m':    sr_base['z_center_m']    if pos_dis else sr_active['z_center_m'],
            'z_half_span_m': sr_base['z_half_span_m'] if pos_dis else sr_active['z_half_span_m'],
            'yaw_abs_rad':   sr_base['yaw_abs_rad']   if yaw_dis else sr_active['yaw_abs_rad'],
        }
        if pos_dis or yaw_dis:
            status_pos = "DISABLED" if pos_dis else "ENABLED"
            status_yaw = "DISABLED" if yaw_dis else "ENABLED"
            task.log_curriculum_update(f"   2. SPAWN RANDOMIZATION: position={status_pos}, orientation={status_yaw}")
        task.log_curriculum_update(
            f"   2. SPAWN: X∈[{(-sr_use['x_half_span_m']):.1f}, {(+sr_use['x_half_span_m']):.1f}] m, "
            f"Y∈[{(sr_use['y_center_m']-sr_use['y_half_span_m']):.1f}, {(sr_use['y_center_m']+sr_use['y_half_span_m']):.1f}] m, "
            f"Z∈[{(sr_use['z_center_m']-sr_use['z_half_span_m']):.1f}, {(sr_use['z_center_m']+sr_use['z_half_span_m']):.1f}] m; yaw ±{(sr_use['yaw_abs_rad']*57.2958):.1f}°"
        )
    except (ValueError, TypeError) as e:
        task.log_curriculum_update(f"   2. SPAWN: (fallback) Using fixed LMF2 config due to: {e}")
    # Get current randomized angle for first environment (representative)
    current_angle = 0.0
    if hasattr(task.static_camera_manager, 'current_camera_angles'):
        current_angle = task.static_camera_manager.current_camera_angles[0] if task.static_camera_manager.current_camera_angles else 0.0
    # Report static camera orientation randomization status (only relevant when yaw sweep is DISABLED)
    try:
        cam_orient_disabled = bool(task.sim_env.global_tensor_dict.get('static_camera_randomization/orientation_disabled', False))
    except (KeyError, TypeError):
        cam_orient_disabled = False
    if yaw_enabled and not dynamic_effective:
        task.log_curriculum_update(f"   4. CAMERA ANGLE: overridden by yaw sweep (env0 current: {current_angle:.1f}°)")
    elif dynamic_effective:
        task.log_curriculum_update(f"   4. CAMERA ANGLE: suppressed (dynamic camera following active)")
    elif cam_orient_disabled:
        task.log_curriculum_update(f"   4. CAMERA ANGLE: randomization DISABLED, fixed at 0.0° (env0: {current_angle:.1f}°)")
    else:
        task.log_curriculum_update(f"   4. CAMERA ANGLE: ±{task.max_camera_angle:.1f}deg max range, env0: {current_angle:.1f}deg (fixed per episode)")
    
    # 4. GATE SIZE UNLOCKS (Curriculum-gated randomization) or Fixed (ablation)
    if hasattr(task.sim_env, 'global_tensor_dict'):
        gate_names = []
        if len(task.sim_env.global_tensor_dict.get("gate_variant_names_per_env", [])) > 0:
            gate_names = task.sim_env.global_tensor_dict["gate_variant_names_per_env"][0]
        # Report fixed mode if enabled
        disable_flag = task.sim_env.global_tensor_dict.get('gate_randomization/disabled', False)
        try:
            if hasattr(disable_flag, 'item'):
                disable_flag = bool(disable_flag.item())
            else:
                disable_flag = bool(disable_flag)
        except Exception:
            disable_flag = False
        if disable_flag:
            try:
                fixed_scale = task.sim_env.global_tensor_dict.get('gate_randomization/fixed_scale_percent', 100)
                if hasattr(fixed_scale, 'item'):
                    fixed_scale = int(fixed_scale.item())
                else:
                    fixed_scale = int(fixed_scale)
            except (ValueError, TypeError):
                fixed_scale = 100
            task.log_curriculum_update(f"   4. GATE SIZE: randomization disabled, fixed scale = {fixed_scale}%")
        else:
            # Compute linear threshold from 80 -> 60 over levels 3..23
            # If EVAL_STRETCH_ENABLED, extend further to 50% by eval_end_level
            stretch_enabled = _os.environ.get("EVAL_STRETCH_ENABLED", "0").strip() in ("1", "true", "True")
            eval_end = int(_os.environ.get("EVAL_STRETCH_END_LEVEL", str(task.task_config.curriculum.eval_stretch_end_level)))
            level = int(task.curriculum_level)
            if level <= 3:
                min_scale = 80
            elif level <= 23:
                frac = (level - 3) / (23 - 3)
                raw = 80 - frac * (80 - 60)
                min_scale = int((int(raw) // 2) * 2)
            elif stretch_enabled:
                # Extend 23->eval_end: 60% -> 50% linearly
                if level >= eval_end:
                    min_scale = 50
                else:
                    extra_frac = (level - 23) / max(1, (eval_end - 23))
                    raw = 60 - extra_frac * (60 - 50)
                    min_scale = int((int(raw) // 2) * 2)
            else:
                min_scale = 60
            if min_scale < 50:
                min_scale = 50
            if min_scale > 100:
                min_scale = 100
            # Collect scales meeting threshold
            scales = []
            for n in gate_names:
                if isinstance(n, str) and "gate_scale_" in n:
                    try:
                        s = int(n.replace("gate_scale_", ""))
                        if s >= min_scale:
                            scales.append(s)
                    except:
                        pass
            # Report unique scales only (avoid duplicates from config classes)
            scales = sorted(list(set(scales)), reverse=True)
            task.log_curriculum_update(f"   4. GATE SIZE: unlocked scales >= {min_scale}% -> {scales if scales else [100]} (uniform across unique scales)")
    
    # 5. CAMERA NOISE PROGRESSION (D455 Simulation)
    camera_gaussian_std, camera_dropout_rate = task.task_config.curriculum.get_camera_noise(task.curriculum_level)
    try:
        cam_noise_disabled = bool(task.sim_env.global_tensor_dict.get('camera_randomization/noise_disabled', False))
    except (KeyError, TypeError):
        cam_noise_disabled = False
    # Per-camera overrides for noise (presence-based overrides)
    try:
        gtd = getattr(task.sim_env, 'global_tensor_dict', {})
        drone_noise_key_present = 'camera_randomization/drone_noise_disabled' in gtd
        static_noise_key_present = 'camera_randomization/static_noise_disabled' in gtd
        drone_noise_flag = bool(gtd.get('camera_randomization/drone_noise_disabled', False)) if drone_noise_key_present else cam_noise_disabled
        static_noise_flag = bool(gtd.get('camera_randomization/static_noise_disabled', False)) if static_noise_key_present else cam_noise_disabled
    except (KeyError, TypeError):
        drone_noise_flag = cam_noise_disabled
        static_noise_flag = cam_noise_disabled
    # Level-3 fallbacks when disabled
    d_std_min, d_drop_min = task.task_config.curriculum.get_camera_noise(3)
    eff_drone_std = camera_gaussian_std if not drone_noise_flag else d_std_min
    eff_static_std = camera_gaussian_std if not static_noise_flag else d_std_min
    eff_drone_drop = camera_dropout_rate if not drone_noise_flag else d_drop_min
    eff_static_drop = camera_dropout_rate if not static_noise_flag else d_drop_min
    task.log_curriculum_update(
        f"   5. CAMERA NOISE: drone(std={eff_drone_std:.4f}, pixel_drop={eff_drone_drop*100:.1f}%), static(std={eff_static_std:.4f}, pixel_drop={eff_static_drop*100:.1f}%)"
    )
    
    # 6. CAMERA FRAME DROPOUT (entire-frame)
    fd = task.task_config.curriculum.get_camera_frame_dropout(task.curriculum_level)
    try:
        cam_fd_disabled = bool(task.sim_env.global_tensor_dict.get('camera_randomization/frame_dropout_disabled', False))
    except (KeyError, TypeError):
        cam_fd_disabled = False
    # Per-camera overrides for frame dropout (presence-based overrides)
    try:
        gtd = getattr(task.sim_env, 'global_tensor_dict', {})
        drone_fd_key_present = 'camera_randomization/drone_frame_dropout_disabled' in gtd
        static_fd_key_present = 'camera_randomization/static_frame_dropout_disabled' in gtd
        drone_fd_flag = bool(gtd.get('camera_randomization/drone_frame_dropout_disabled', False)) if drone_fd_key_present else cam_fd_disabled
        static_fd_flag = bool(gtd.get('camera_randomization/static_frame_dropout_disabled', False)) if static_fd_key_present else cam_fd_disabled
    except (KeyError, TypeError):
        drone_fd_flag = cam_fd_disabled
        static_fd_flag = cam_fd_disabled
    # After change: when disabled, show level-3 minimum totals instead of 0
    fd_min = task.task_config.curriculum.get_camera_frame_dropout(3)
    eff_drone_tot = fd['drone_total'] if not drone_fd_flag else fd_min['drone_total']
    eff_static_tot = fd['static_total'] if not static_fd_flag else fd_min['static_total']
    eff_drone_freeze = fd['drone_freeze'] if not drone_fd_flag else fd_min['drone_freeze']
    eff_drone_blank = fd['drone_blank'] if not drone_fd_flag else fd_min['drone_blank']
    eff_static_freeze = fd['static_freeze'] if not static_fd_flag else fd_min['static_freeze']
    eff_static_blank = fd['static_blank'] if not static_fd_flag else fd_min['static_blank']
    task.log_curriculum_update(
        f"   6. CAMERA FRAME DROPOUT: drone_total={eff_drone_tot*100:.1f}% (freeze {eff_drone_freeze*100:.1f}%, blank {eff_drone_blank*100:.1f}%), static_total={eff_static_tot*100:.1f}% (freeze {eff_static_freeze*100:.1f}%, blank {eff_static_blank*100:.1f}%)"
    )
    
    # 7. STATE NOISE (pose)
    if task.task_config.curriculum.enable_state_noise:
        try:
            state_noise_disabled = bool(task.sim_env.global_tensor_dict.get('state_randomization/noise_disabled', False))
        except (KeyError, TypeError):
            state_noise_disabled = False
        if state_noise_disabled:
            task.log_curriculum_update("   7. STATE NOISE: DISABLED (all std=0)")
        else:
            sn = task.task_config.curriculum.get_state_noise(task.curriculum_level)
            task.log_curriculum_update(
                f"   7. STATE NOISE: drone_pos_std={sn['drone_pos_std_m']:.4f} m, drone_orient_std={sn['drone_orient_std_rad']*57.2958:.3f} deg, "
                f"static_pos_std={sn['static_pos_std_m']:.4f} m, static_orient_std={sn['static_orient_std_rad']*57.2958:.3f} deg"
            )
    else:
        task.log_curriculum_update("   7. STATE NOISE: disabled")
    
    # 8. DYNAMIC CAMERA FOLLOWING
    dynamic_enabled = task.task_config.curriculum.enable_dynamic_camera_following
    try:
        dynamic_disabled = bool(task.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False))
        config_overridden = bool(task.sim_env.global_tensor_dict.get('dynamic_camera_following/config_overridden', False))
    except (KeyError, TypeError):
        dynamic_disabled = False
        config_overridden = False
    
    if dynamic_enabled and not dynamic_disabled:
        if config_overridden:
            task.log_curriculum_update("   8. DYNAMIC CAMERA: ENABLED by flag (--enable_dynamic_camera_following=true) - camera follows drone with adaptive gate targeting")
        else:
            task.log_curriculum_update("   8. DYNAMIC CAMERA: ENABLED (camera follows drone with adaptive gate targeting)")
    elif dynamic_enabled and dynamic_disabled:
        task.log_curriculum_update("   8. DYNAMIC CAMERA: DISABLED by flag (--disable_dynamic_camera_following=true)")
    else:
        if config_overridden:
            task.log_curriculum_update("   8. DYNAMIC CAMERA: DISABLED by flag (--enable_dynamic_camera_following=false)")
        else:
            task.log_curriculum_update("   8. DYNAMIC CAMERA: DISABLED (static camera mode - curriculum-based positioning)")
    
    # Curriculum multiplier debug (update block)
    cm_disabled = read_env_bool("SF_DISABLE_CURRICULUM_MULTIPLIER", task.task_config.disable_curriculum_multiplier)
    if not cm_disabled:
        cm_disabled = bool(task.task_config.disable_curriculum_multiplier)
    frac_eff = 0.0 if cm_disabled else float(task.curriculum_progress_fraction)
    factor = 1.0 + 0.5 * frac_eff
    task.log_curriculum_update(f"   8. CURRICULUM MULTIPLIER: {'DISABLED' if cm_disabled else 'ENABLED'} (factor={factor:.3f})")
    
    task.log_curriculum_update(f"[CURRICULUM UPDATE] FINAL STATE:")
    task.log_curriculum_update(f"[CURRICULUM UPDATE]   Level: {task.curriculum_level} (range: {task.task_config.curriculum.min_level}-{task.task_config.curriculum.max_level})")
    task.log_curriculum_update(f"[CURRICULUM UPDATE]   Max level reached: {task.max_curriculum_level_reached} (DECREASE ENABLED)")
    task.log_curriculum_update(f"[CURRICULUM UPDATE]   Progress: {task.curriculum_progress_fraction:.3f}")
    task.log_curriculum_update(f"[CURRICULUM UPDATE]   Obstacles behind gate: {obstacles_behind_gate} (total assets: {total_obstacles_in_env} = {fixed_assets_visible} visible + {obstacles_behind_gate} curriculum)")
    task.log_curriculum_update(f"[CURRICULUM UPDATE]   Asset manager: Updated both obs_dict and global_tensor_dict with count {total_obstacles_in_env}")
    # Report spawn ablation status with effective ranges
    try:
        baseline_level = int(task.task_config.curriculum.min_level)
        pos_dis = bool(task.sim_env.global_tensor_dict.get('spawn_randomization/position_disabled', False))
        yaw_dis = bool(task.sim_env.global_tensor_dict.get('spawn_randomization/orientation_disabled', False))
        sr_active = task.task_config.curriculum.get_spawn_ranges(task.curriculum_level)
        sr_base = task.task_config.curriculum.get_spawn_ranges(baseline_level)
        sr_use = {
            'x_half_span_m': sr_base['x_half_span_m'] if pos_dis else sr_active['x_half_span_m'],
            'y_center_m':    sr_base['y_center_m']    if pos_dis else sr_active['y_center_m'],
            'y_half_span_m': sr_base['y_half_span_m'] if pos_dis else sr_active['y_half_span_m'],
            'z_center_m':    sr_base['z_center_m']    if pos_dis else sr_active['z_center_m'],
            'z_half_span_m': sr_base['z_half_span_m'] if pos_dis else sr_active['z_half_span_m'],
            'yaw_abs_rad':   sr_base['yaw_abs_rad']   if yaw_dis else sr_active['yaw_abs_rad'],
        }
        status_pos = "DISABLED" if pos_dis else "ENABLED"
        status_yaw = "DISABLED" if yaw_dis else "ENABLED"
        task.log_curriculum_update(f"[CURRICULUM UPDATE]   Spawn randomization: position={status_pos}, orientation={status_yaw}")
        task.log_curriculum_update(
            f"[CURRICULUM UPDATE]   Spawn ranges: X∈[{(-sr_use['x_half_span_m']):.1f}, {(+sr_use['x_half_span_m']):.1f}] m, "
            f"Y∈[{(sr_use['y_center_m']-sr_use['y_half_span_m']):.1f}, {(sr_use['y_center_m']+sr_use['y_half_span_m']):.1f}] m, "
            f"Z∈[{(sr_use['z_center_m']-sr_use['z_half_span_m']):.1f}, {(sr_use['z_center_m']+sr_use['z_half_span_m']):.1f}] m; yaw ±{(sr_use['yaw_abs_rad']*57.2958):.1f}°"
        )
    except (ValueError, TypeError):
        task.log_curriculum_update(f"[CURRICULUM UPDATE]   Spawn difficulty: LMF2 config (fallback)")
    # When yaw sweep is enabled and dynamic camera is not active, suppress static camera angle randomization message
    try:
        yaw_enabled = str(os.environ.get('SF_ENABLE_STATIC_CAMERA_YAW_SWEEP', 'false')).lower() == 'true'
    except (KeyError, TypeError):
        yaw_enabled = False
    dynamic_effective = False
    try:
        dyn_cfg = task.task_config.curriculum.enable_dynamic_camera_following
        dyn_dis = bool(task.sim_env.global_tensor_dict.get('dynamic_camera_following/disabled', False))
        dynamic_effective = bool(dyn_cfg and not dyn_dis)
    except (KeyError, TypeError):
        dynamic_effective = False
    if yaw_enabled and not dynamic_effective:
        # Already logged as overridden by yaw sweep earlier
        pass
    else:
        task.log_curriculum_update(f"[CURRICULUM UPDATE]   Camera angle: ±{task.max_camera_angle:.1f}deg max range (randomized per episode reset, fixed during episode)")

def _populate_curriculum_infos(
    self,
    success_rate: float,
    crash_rate: float,
    timeout_rate: float,
    obstacles_behind_gate: int,
    total_obstacles_in_env: int,
) -> None:
    """Populate task.infos with curriculum metrics for wandb logging."""
    # Add comprehensive curriculum metrics to infos for wandb logging
    task.infos["curriculum/level"] = torch.as_tensor(task.curriculum_level, dtype=torch.float32)
    task.infos["curriculum/progress"] = torch.as_tensor(task.curriculum_progress_fraction, dtype=torch.float32)
    task.infos["curriculum/success_rate"] = torch.as_tensor(success_rate, dtype=torch.float32)
    task.infos["curriculum/crash_rate"] = torch.as_tensor(crash_rate, dtype=torch.float32)
    task.infos["curriculum/timeout_rate"] = torch.as_tensor(timeout_rate, dtype=torch.float32)
    
    # Add curriculum metrics
    task.infos["curriculum/obstacles_behind_gate"] = torch.as_tensor(obstacles_behind_gate, dtype=torch.float32)
    task.infos["curriculum/total_assets"] = torch.as_tensor(total_obstacles_in_env, dtype=torch.float32)
    task.infos["curriculum/max_level_reached"] = torch.as_tensor(task.max_curriculum_level_reached, dtype=torch.float32)
    
    # Add camera noise metrics (D455 simulation) — report effective per-camera values in logs above
    task.infos["curriculum/camera_gaussian_std"] = torch.as_tensor(camera_gaussian_std, dtype=torch.float32)
    task.infos["curriculum/camera_dropout_rate"] = torch.as_tensor(camera_dropout_rate, dtype=torch.float32)
    # Per-camera effective values (respecting per-camera disable overrides) — level-3 fallback when disabled
    gtd = getattr(task.sim_env, 'global_tensor_dict', {})
    drone_noise_dis = bool(gtd.get('camera_randomization/drone_noise_disabled', False))
    static_noise_dis = bool(gtd.get('camera_randomization/static_noise_disabled', False))
    # Level-3 minimums
    d_std_min, d_drop_min = task.task_config.curriculum.get_camera_noise(3)
    # Effective Gaussian std per camera
    eff_gauss_drone = camera_gaussian_std if not drone_noise_dis else d_std_min
    eff_gauss_static = camera_gaussian_std if not static_noise_dis else d_std_min
    # Effective pixel dropout per camera
    eff_drop_drone = camera_dropout_rate if not drone_noise_dis else d_drop_min
    eff_drop_static = camera_dropout_rate if not static_noise_dis else d_drop_min
    task.infos["curriculum/camera_noise_drone_gaussian_std"] = torch.tensor(eff_gauss_drone, dtype=torch.float32)
    task.infos["curriculum/camera_noise_static_gaussian_std"] = torch.tensor(eff_gauss_static, dtype=torch.float32)
    task.infos["curriculum/camera_noise_drone_dropout_rate"] = torch.tensor(eff_drop_drone, dtype=torch.float32)
    task.infos["curriculum/camera_noise_static_dropout_rate"] = torch.tensor(eff_drop_static, dtype=torch.float32)
    # Add camera frame dropout metrics (effective per-camera, with level-3 fallback when disabled)
    fd_sched = task.task_config.curriculum.get_camera_frame_dropout(task.curriculum_level)
    try:
        gtd = getattr(task.sim_env, 'global_tensor_dict', {})
        drone_fd_flag = bool(gtd.get('camera_randomization/drone_frame_dropout_disabled', False))
        static_fd_flag = bool(gtd.get('camera_randomization/static_frame_dropout_disabled', False))
    except (KeyError, TypeError):
        drone_fd_flag = False
        static_fd_flag = False
    fd_min = task.task_config.curriculum.get_camera_frame_dropout(3)
    eff = {
        'drone_total':  fd_sched['drone_total']  if not drone_fd_flag else fd_min['drone_total'],
        'static_total': fd_sched['static_total'] if not static_fd_flag else fd_min['static_total'],
        'drone_freeze': fd_sched['drone_freeze'] if not drone_fd_flag else fd_min['drone_freeze'],
        'drone_blank':  fd_sched['drone_blank']  if not drone_fd_flag else fd_min['drone_blank'],
        'static_freeze':fd_sched['static_freeze']if not static_fd_flag else fd_min['static_freeze'],
        'static_blank': fd_sched['static_blank'] if not static_fd_flag else fd_min['static_blank'],
    }
    task.infos["curriculum/camera_frame_dropout_drone_total"] = torch.tensor(eff["drone_total"], dtype=torch.float32)
    task.infos["curriculum/camera_frame_dropout_static_total"] = torch.tensor(eff["static_total"], dtype=torch.float32)
    task.infos["curriculum/camera_frame_freeze_drone"] = torch.tensor(eff["drone_freeze"], dtype=torch.float32)
    task.infos["curriculum/camera_frame_blank_drone"] = torch.tensor(eff["drone_blank"], dtype=torch.float32)
    task.infos["curriculum/camera_frame_freeze_static"] = torch.tensor(eff["static_freeze"], dtype=torch.float32)
    task.infos["curriculum/camera_frame_blank_static"] = torch.tensor(eff["static_blank"], dtype=torch.float32)
    
    # Add camera angle metrics
    task.infos["curriculum/camera_max_angle"] = torch.tensor(task.max_camera_angle, dtype=torch.float32)
    # Use first environment's angle as representative for wandb tracking
    current_angle = 0.0
    if hasattr(task.static_camera_manager, 'current_camera_angles'):
        current_angle = task.static_camera_manager.current_camera_angles[0] if task.static_camera_manager.current_camera_angles else 0.0
    task.infos["curriculum/camera_current_angle"] = torch.tensor(current_angle, dtype=torch.float32)
    # Track ablation flag in infos
    try:
        cam_orient_disabled = bool(task.sim_env.global_tensor_dict.get('static_camera_randomization/orientation_disabled', False))
    except (KeyError, TypeError):
        cam_orient_disabled = False
    task.infos["curriculum/camera_orientation_randomization_disabled"] = torch.tensor(1.0 if cam_orient_disabled else 0.0, dtype=torch.float32)
    
    # Add state noise metrics
    if task.task_config.curriculum.enable_state_noise:
        sn = task.task_config.curriculum.get_state_noise(task.curriculum_level)
        task.infos["curriculum/state_noise_drone_pos_std_m"] = torch.tensor(sn["drone_pos_std_m"], dtype=torch.float32)
        task.infos["curriculum/state_noise_drone_orient_std_deg"] = torch.tensor(sn["drone_orient_std_rad"]*57.2958, dtype=torch.float32)
        task.infos["curriculum/state_noise_static_pos_std_m"] = torch.tensor(sn["static_pos_std_m"], dtype=torch.float32)
        task.infos["curriculum/state_noise_static_orient_std_deg"] = torch.tensor(sn["static_orient_std_rad"]*57.2958, dtype=torch.float32)

def update_episode_reward_tracking(task, obs_dict: dict[str, torch.Tensor], rewards: torch.Tensor, crashes: torch.Tensor) -> None:
    """Update cumulative episode reward tracking for comprehensive debugging."""
    robot_position = obs_dict["robot_position"]
    
    # Calculate individual reward components (same as in compute_rewards_and_crashes)
    dist = torch.norm(task.pos_error_vehicle_frame, dim=1)
    prev_dist = torch.norm(task.pos_error_vehicle_frame_prev, dim=1)
    action = obs_dict["robot_actions"].clone()
    prev_action = obs_dict["robot_prev_actions"].clone()
    
    mult_factor = 1.0 + (0.5) * task.curriculum_progress_fraction

    # Position reward
    pos_reward = exponential_reward_function(
        task.task_config.reward_parameters["pos_reward_magnitude"],
        task.task_config.reward_parameters["pos_reward_exponent"],
        dist,
    )
    task.episode_pos_reward += mult_factor * pos_reward
    
    # Very close reward
    very_close_reward = exponential_reward_function(
        task.task_config.reward_parameters["very_close_to_goal_reward_magnitude"],
        task.task_config.reward_parameters["very_close_to_goal_reward_exponent"],
        dist,
    )
    task.episode_very_close_reward += mult_factor * very_close_reward
    
    # Getting closer reward
    getting_closer = prev_dist - dist
    getting_closer_reward = torch.where(
        getting_closer > 0,
        task.task_config.reward_parameters["getting_closer_reward_multiplier"] * getting_closer,
        2.0 * task.task_config.reward_parameters["getting_closer_reward_multiplier"] * getting_closer,
    )
    task.episode_getting_closer_reward += mult_factor * getting_closer_reward
    
    # Gate approach reward
    # Use adaptive gate center (z = bottom + center_height)
    gate_center_position = task.gate_position.clone()
    gate_center_position[:, 2] = gate_center_position[:, 2] + task.gate_center_height
    gate_distance = torch.norm(robot_position - gate_center_position, dim=1)
    gate_approach_reward = exponential_reward_function(
        task.task_config.reward_parameters["gate_approach_reward_magnitude"],
        0.5,
        gate_distance,
    )
    task.episode_gate_approach_reward += mult_factor * gate_approach_reward
    
    # Gate alignment reward
    gate_alignment_reward = torch.zeros_like(gate_distance)
    aligned_mask = torch.abs(robot_position[:, 0] - task.gate_position[:, 0]) < 1.5
    gate_alignment_reward[aligned_mask] = task.task_config.reward_parameters["gate_alignment_reward_magnitude"]
    task.episode_gate_alignment_reward += mult_factor * gate_alignment_reward
    
    # Camera facing reward (same calculation as in debugging section)
    robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
    drone_to_gate = task.gate_position - robot_position
    drone_to_gate_normalized = drone_to_gate / (torch.norm(drone_to_gate, dim=1, keepdim=True) + 1e-8)
    
    # Get drone's forward direction (where camera points)
    qw, qx, qy, qz = robot_vehicle_orientation[:, 3], robot_vehicle_orientation[:, 0], robot_vehicle_orientation[:, 1], robot_vehicle_orientation[:, 2]
    forward_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    forward_y = 2.0 * (qx * qy + qw * qz)
    forward_z = 2.0 * (qx * qz - qw * qy)
    drone_forward = torch.stack([forward_x, forward_y, forward_z], dim=1)
    drone_forward_normalized = drone_forward / (torch.norm(drone_forward, dim=1, keepdim=True) + 1e-8)
    
    # Calculate alignment and camera facing reward
    camera_gate_alignment = torch.sum(drone_forward_normalized * drone_to_gate_normalized, dim=1)
    camera_gate_alignment = torch.clamp(camera_gate_alignment, -1.0, 1.0)
    
    camera_facing_reward = torch.zeros_like(camera_gate_alignment)
    perfect_mask = camera_gate_alignment > 0.966
    camera_facing_reward[perfect_mask] = task.task_config.reward_parameters["camera_facing_reward_magnitude"]
    excellent_mask = (camera_gate_alignment > 0.866) & (camera_gate_alignment <= 0.966)
    camera_facing_reward[excellent_mask] = 0.9 * task.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[excellent_mask]
    good_mask = (camera_gate_alignment > 0.5) & (camera_gate_alignment <= 0.866)
    camera_facing_reward[good_mask] = 0.8 * task.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[good_mask]
    moderate_mask = (camera_gate_alignment > 0.0) & (camera_gate_alignment <= 0.5)
    camera_facing_reward[moderate_mask] = 0.4 * task.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[moderate_mask]
    poor_mask = (camera_gate_alignment > -0.707) & (camera_gate_alignment <= 0.0)
    camera_facing_reward[poor_mask] = 0.2 * task.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[poor_mask]
    severe_mask = camera_gate_alignment <= -0.707
    camera_facing_reward[severe_mask] = 2.0 * task.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[severe_mask]
    # Gate the camera-facing reward: only before first crossing and while approaching (y below gate plane)
    approach_mask = (robot_position[:, 1] < task.gate_position[:, 1] - 0.1) & (~task.gate_passed)
    camera_facing_reward = camera_facing_reward * approach_mask.float()
    task.episode_camera_facing_reward += mult_factor * camera_facing_reward
    
    # Action penalties - FIXED: Added missing Y-action penalties for 4D action space  
    action_diff = action - prev_action
    
    x_diff_penalty = exponential_penalty_function(
        task.task_config.reward_parameters["x_action_diff_penalty_magnitude"],
        task.task_config.reward_parameters["x_action_diff_penalty_exponent"],
        action_diff[:, 0],
    )
    y_diff_penalty = exponential_penalty_function(
        task.task_config.reward_parameters["y_action_diff_penalty_magnitude"],
        task.task_config.reward_parameters["y_action_diff_penalty_exponent"],
        action_diff[:, 1],
    )
    z_diff_penalty = exponential_penalty_function(
        task.task_config.reward_parameters["z_action_diff_penalty_magnitude"],
        task.task_config.reward_parameters["z_action_diff_penalty_exponent"],
        action_diff[:, 2],
    )
    yawrate_diff_penalty = exponential_penalty_function(
        task.task_config.reward_parameters["yawrate_action_diff_penalty_magnitude"],
        task.task_config.reward_parameters["yawrate_action_diff_penalty_exponent"],
        action_diff[:, 3],
    )
    
    x_absolute_penalty = exponential_penalty_function(
        task.task_config.reward_parameters["x_absolute_action_penalty_magnitude"],
        task.task_config.reward_parameters["x_absolute_action_penalty_exponent"],
        action[:, 0],
    )
    y_absolute_penalty = exponential_penalty_function(
        task.task_config.reward_parameters["y_absolute_action_penalty_magnitude"],
        task.task_config.reward_parameters["y_absolute_action_penalty_exponent"],
        action[:, 1],
    )
    z_absolute_penalty = exponential_penalty_function(
        task.task_config.reward_parameters["z_absolute_action_penalty_magnitude"],
        task.task_config.reward_parameters["z_absolute_action_penalty_exponent"],
        action[:, 2],
    )
    yawrate_absolute_penalty = exponential_penalty_function(
        task.task_config.reward_parameters["yawrate_absolute_action_penalty_magnitude"],
        task.task_config.reward_parameters["yawrate_absolute_action_penalty_exponent"],
        action[:, 3],
    )
    
    action_diff_penalty = x_diff_penalty + y_diff_penalty + z_diff_penalty + yawrate_diff_penalty
    absolute_action_penalty = x_absolute_penalty + y_absolute_penalty + z_absolute_penalty + yawrate_absolute_penalty
    total_action_penalty = action_diff_penalty + absolute_action_penalty
    task.episode_action_penalty += total_action_penalty
    
    # Track collision penalties
    collision_mask = crashes > 0
    collision_penalty = torch.where(
        collision_mask,
        task.task_config.reward_parameters["collision_penalty"],
        torch.zeros_like(crashes, dtype=torch.float32),
    )
    task.episode_collision_penalty += collision_penalty
    
    # Track gate passage rewards (check if any gate passages occurred this step) - ADAPTIVE
    # Use the same logic as main reward system with adaptive dimensions
    tracking_width_tolerance = task.gate_width * 0.6  # 60% of gate width
    tracking_min_height = task.gate_position[:, 2] + task.gate_height * 0.08  # 8% above ground
    tracking_max_height = task.gate_position[:, 2] + task.gate_height * 0.92  # 92% of gate height
    
    gate_passed_this_step = (
        (robot_position[:, 1] > task.gate_position[:, 1]) &
        (torch.abs(robot_position[:, 0] - task.gate_position[:, 0]) < tracking_width_tolerance) &
        (robot_position[:, 2] > tracking_min_height) & (robot_position[:, 2] < tracking_max_height) &
        (~task.gate_passed)  # Haven't passed before
    )
    
    # Center passage detection with adaptive dimensions (like main system)
    x_distance_from_center = torch.abs(robot_position[:, 0] - task.gate_position[:, 0])
    z_distance_from_center = torch.abs(robot_position[:, 2] - (task.gate_position[:, 2] + task.gate_center_height))
    
    # Adaptive center thresholds
    center_x_threshold = task.gate_width * 0.2  # 20% of gate width for center alignment
    center_z_threshold = task.gate_height * 0.125  # 12.5% of gate height for center alignment
    center_aligned_mask = (x_distance_from_center < center_x_threshold) & (z_distance_from_center < center_z_threshold)
    
    # Basic gate passage reward
    gate_passage_reward = torch.where(
        gate_passed_this_step,
        mult_factor * task.task_config.reward_parameters["gate_passage_reward_magnitude"],
        torch.zeros_like(gate_passed_this_step, dtype=torch.float32),
    )
    
    # Gate center passage bonus (only for centered passages)
    gate_center_passage_bonus = torch.where(
        gate_passed_this_step & center_aligned_mask,
        mult_factor * task.task_config.reward_parameters["gate_center_passage_bonus_magnitude"],
        torch.zeros_like(gate_passed_this_step, dtype=torch.float32),
    )
    
    task.gate_passed = task.gate_passed | gate_passed_this_step
    
    # Track total gate passage rewards (basic + center bonus)
    total_gate_rewards = gate_passage_reward + gate_center_passage_bonus
    task.episode_gate_passage_reward += total_gate_rewards
    
    # Boundary violation penalty (episode tracking mirror of TorchScript path)
    y_margin = 0.2
    behind_gate_mask = (robot_position[:, 1] > (task.gate_position[:, 1] + y_margin))
    full_width_tol = task.gate_width * 0.5
    full_min_h = task.gate_position[:, 2] + 0.0 * task.gate_height
    full_max_h = task.gate_position[:, 2] + 1.0 * task.gate_height
    within_passage_window = (
        (torch.abs(robot_position[:, 0] - task.gate_position[:, 0]) < full_width_tol)
        & (robot_position[:, 2] > full_min_h)
        & (robot_position[:, 2] < full_max_h)
    )
    misaligned_cross_mask = behind_gate_mask & (~within_passage_window) & (~task.gate_passed) & (~gate_passed_this_step)
    boundary_violation_penalty = torch.zeros_like(gate_distance)
    # One-shot penalty per episode: apply only on rising edge
    if not True:
        task._bv_flag_episode = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
    rising_mask = misaligned_cross_mask & (~task._bv_flag_episode)
    boundary_violation_penalty[rising_mask] = -50.0
    # Update flag and force termination for violating envs
    task._bv_flag_episode = task._bv_flag_episode | misaligned_cross_mask
    if torch.any(rising_mask):
        # Use terminations for boundary violations (true MDP failure)
        task.terminations[rising_mask] = 1
        # env0 debug already prints above on rising edge
    task.episode_boundary_violation_penalty += boundary_violation_penalty
    # Print on rising edge for all envs that violated this step
    rising_envs = torch.nonzero(rising_mask, as_tuple=False).squeeze(-1)
    if rising_envs.numel() > 0:
        for eid in rising_envs.tolist():
            rx = float(robot_position[eid, 0].item())
            ry = float(robot_position[eid, 1].item())
            rz = float(robot_position[eid, 2].item())
            gate_x = float(task.gate_position[eid, 0].item())
            gate_y = float(task.gate_position[eid, 1].item())
            x_off = abs(rx - gate_x)
            gw = float(task.gate_width[eid].item() if hasattr(task.gate_width, 'shape') else task.gate_width)
            gh = float(task.gate_height[eid].item() if hasattr(task.gate_height, 'shape') else task.gate_height)
            tol = float(full_width_tol[eid].item() if hasattr(full_width_tol, 'shape') else full_width_tol)
            zmin = float(full_min_h[eid].item() if hasattr(full_min_h, 'shape') else full_min_h)
            zmax = float(full_max_h[eid].item() if hasattr(full_max_h, 'shape') else full_max_h)
            logger.warning(
                f"[Boundary] Env{eid} VIOLATION at step {task.num_task_steps}: pos=({rx:.3f},{ry:.3f},{rz:.3f}), "
                f"gate_y={gate_y:.3f}, x_off={x_off:.3f} (tol={tol:.3f}), z_window=({zmin:.3f},{zmax:.3f}), "
                f"gate_size=(w={gw:.3f}, h={gh:.3f})"
            )
    
    # Track image rewards (from post_image_reward_addition)
    if task.min_pixel_dist is not None:
        mag = task.task_config.reward_parameters.get("image_penalty_magnitude", 4.0)
        expo = task.task_config.reward_parameters.get("image_penalty_exponent", 1.0)
        image_rewards = -exponential_reward_function(
            float(mag), float(expo), task.min_pixel_dist[~task.terminations]
        )
        # Only add for non-terminated environments
        non_terminated_mask = ~task.terminations
        if torch.sum(non_terminated_mask) > 0:
            task.episode_image_reward[non_terminated_mask] += image_rewards
    
    # Increment episode length tracking
    task.episode_lengths += 1

def reset_episode_reward_tracking(task, env_ids: torch.Tensor) -> None:
    """Reset episode reward tracking for specified environments when episodes end."""
    if len(env_ids) == 0:
        return
        
    # Store completed episode data for averaging
    for env_id in env_ids:
        if task.episode_lengths[env_id] > 0:  # Valid episode
            episode_data = {
                'total_reward': (
                    task.episode_pos_reward[env_id] + 
                    task.episode_very_close_reward[env_id] + 
                    task.episode_getting_closer_reward[env_id] + 
                    task.episode_gate_approach_reward[env_id] + 
                    task.episode_gate_alignment_reward[env_id] + 
                    task.episode_camera_facing_reward[env_id] + 
                    task.episode_action_penalty[env_id] + 
                    task.episode_time_penalty[env_id] + 
                    task.episode_timeout_penalty[env_id] + 
                    task.episode_boundary_violation_penalty[env_id] + 
                    task.episode_gate_passage_reward[env_id] + 
                    task.episode_collision_penalty[env_id] + 
                    task.episode_image_reward[env_id]
                ).item(),
                'pos_reward': task.episode_pos_reward[env_id].item(),
                'very_close_reward': task.episode_very_close_reward[env_id].item(),
                'getting_closer_reward': task.episode_getting_closer_reward[env_id].item(),
                'gate_approach_reward': task.episode_gate_approach_reward[env_id].item(),
                'gate_alignment_reward': task.episode_gate_alignment_reward[env_id].item(),
                'camera_facing_reward': task.episode_camera_facing_reward[env_id].item(),
                'action_penalty': task.episode_action_penalty[env_id].item(),
                'boundary_violation_penalty': task.episode_boundary_violation_penalty[env_id].item(),
                'time_penalty': task.episode_time_penalty[env_id].item(),
                'timeout_penalty': task.episode_timeout_penalty[env_id].item(),
                'gate_passage_reward': task.episode_gate_passage_reward[env_id].item(),  # Now includes both basic + center bonus
                'collision_penalty': task.episode_collision_penalty[env_id].item(),
                'image_reward': task.episode_image_reward[env_id].item(),
                'episode_length': task.episode_lengths[env_id].item(),
            }
            task.completed_episodes.append(episode_data)
            
            # Keep only last N episodes
            if len(task.completed_episodes) > task.max_stored_episodes:
                task.completed_episodes.pop(0)
    
    # Reset trackers for completed episodes
    task.episode_rewards.reset_envs(env_ids)

