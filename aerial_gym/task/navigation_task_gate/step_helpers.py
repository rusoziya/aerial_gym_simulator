from __future__ import annotations

import torch

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import *  # noqa: F401,F403

logger = CustomLogger("navigation_task_gate_step")


class StepHelpers:
    def __init__(self, task: object) -> None:
        self.task = task

    def _validate_and_step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform actions, run physics step, detect NaN/Inf in actions and observations."""
        transformed_action = self.task.action_transformation_function(actions)
        # Action NaN/Inf guard
        try:
            invalid_action_mask = torch.any(torch.isnan(transformed_action) | torch.isinf(transformed_action), dim=1)
            if torch.any(invalid_action_mask):
                transformed_action[invalid_action_mask] = 0.0
                nan_trunc_mask = invalid_action_mask.clone()
                if self.task.task_config.guard_debug_enabled:
                    _ids = torch.nonzero(invalid_action_mask, as_tuple=False).squeeze(-1).tolist()
                    logger.warning(f"[NaNGuard] Invalid ACTION in envs {_ids}; zeroed and will truncate")
            else:
                nan_trunc_mask = torch.zeros(self.task.num_envs, dtype=torch.bool, device=self.task.device)
        except RuntimeError:
            nan_trunc_mask = torch.zeros(self.task.num_envs, dtype=torch.bool, device=self.task.device)

        self.task.sim_env.step(actions=transformed_action)

        # Observation NaN/Inf guard
        for k, v in self.task.obs_dict.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == self.task.num_envs:
                bad = torch.any(torch.isnan(v) | torch.isinf(v), dim=tuple(range(1, v.ndim)))
                if self.task.task_config.guard_debug_enabled and torch.any(bad):
                    _ids = torch.nonzero(bad, as_tuple=False).squeeze(-1).tolist()
                    logger.warning(f"[NaNGuard] Invalid OBS '{k}' in envs {_ids}")
                nan_trunc_mask |= bad
        if torch.any(nan_trunc_mask):
            if "robot_position" in self.task.obs_dict and isinstance(self.task.obs_dict["robot_position"], torch.Tensor):
                self.task.obs_dict["robot_position"][nan_trunc_mask] = 0.0

        return transformed_action, nan_trunc_mask

    def _detect_gate_passage(
        self,
        robot_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect gate passage success and compute immediate/target success masks.

        Returns:
            successes: per-env bool tensor -- gate passage without crash.
            target_successes: per-env bool tensor -- gate passage within 10% center window.
            gate_passage_success: raw gate-passage bool tensor (before crash filtering).
        """
        # Gate passage detection: crossed gate plane within the FULL gate opening (100% tolerance)
        # Accept any passage through the opening: width +/-50% and height from bottom to top
        gate_success_width_tolerance = self.task.gate_width * 0.50
        gate_success_min_height = self.task.gate_position[:, 2]  # gate bottom
        gate_success_max_height = self.task.gate_position[:, 2] + self.task.gate_height  # gate top

        gate_passage_success = (
            (robot_position[:, 1] > self.task.gate_position[:, 1])
            & (torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0]) < gate_success_width_tolerance)
            & (robot_position[:, 2] > gate_success_min_height)
            & (robot_position[:, 2] < gate_success_max_height)
        )

        # Immediate success termination and reset
        # Target window: within 10% of gate width (X) and 10% of gate height (Z) around ADAPTIVE gate center
        x_off_imm = torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0])
        z_off_imm = torch.abs(robot_position[:, 2] - (self.task.gate_position[:, 2] + self.task.gate_center_height))
        x_ok_imm = x_off_imm <= (self.task.gate_width * 0.10)
        z_ok_imm = z_off_imm <= (self.task.gate_height * 0.10)
        target_success_immediate = x_ok_imm & z_ok_imm
        # Do not let target-window success terminate episodes; training success uses gate passage only
        immediate_success_mask = (~(self.task.terminations > 0)) & gate_passage_success
        if torch.any(immediate_success_mask):
            # Mark terminations immediately so the environment will reset at post_reward_calculation_step
            self.task.terminations[immediate_success_mask] = 1
            # Record per-episode target success flags where 10% tolerance is also met
            self.task._ep_target_success_flag[immediate_success_mask] |= target_success_immediate[immediate_success_mask]
            try:
                success_envs = torch.nonzero(immediate_success_mask, as_tuple=False).squeeze(-1).tolist()
            except RuntimeError:
                success_envs = []
            logger.debug(f"[SUCCESS_RESET] Immediate success achieved in envs: {success_envs}. Terminating and resetting.")

        # Success when episode TERMINATES (not crashes) and gate passage achieved
        crash_mask = (self.task.obs_dict["crashes"] > 0)
        successes = (self.task.terminations > 0) & gate_passage_success & (~crash_mask)

        # Target success at truncation: same 10% width/height window around adaptive gate center
        x_off = torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0])
        z_off = torch.abs(robot_position[:, 2] - (self.task.gate_position[:, 2] + self.task.gate_center_height))
        x_ok = x_off <= (self.task.gate_width * 0.10)
        z_ok = z_off <= (self.task.gate_height * 0.10)
        target_success = x_ok & z_ok
        target_successes = (self.task.terminations > 0) & (target_success & gate_passage_success) & (~crash_mask)
        # Also accumulate per-episode target success flag when truncated at step end
        end_success_mask = (self.task.terminations > 0) & (target_success & gate_passage_success) & (~crash_mask)
        self.task._ep_target_success_flag[end_success_mask] = True

        return successes, target_successes, gate_passage_success

    def _apply_timeout_and_populate_infos(
        self,
        successes: torch.Tensor,
    ) -> None:
        """Compute timeout flags, populate self.task.infos, and apply timeout penalty to rewards."""
        timeouts = torch.where(
            self.task.truncations > 0, torch.logical_not(successes), torch.zeros_like(successes)
        )
        timeouts = torch.where(
            self.task.terminations > 0, torch.zeros_like(timeouts), timeouts
        )  # timeouts are not counted if there is a crash

        self.task.infos["successes"] = successes
        self.task.infos["timeouts"] = timeouts
        # Report crashes only (exclude success-based terminations)
        self.task.infos["crashes"] = self.task.obs_dict["crashes"]

        # One-off timeout penalty: discourage hover-to-horizon strategies
        try:
            timeout_penalty = float(self.task.task_config.reward_parameters.get('timeout_penalty', 70.0))
        except (ValueError, TypeError):
            timeout_penalty = 75.0
        if torch.any(timeouts):
            # Apply to the per-env reward vector maintained at the task level
            self.task.rewards = self.task.rewards - (timeouts.float() * timeout_penalty)
            self.task.episode_timeout_penalty[timeouts] -= timeout_penalty

    def _compute_gate_navigation_metrics(
        self,
        robot_position: torch.Tensor,
        camera_gate_alignment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate gate navigation metrics and populate self.task.infos with tracking data.

        Returns:
            gate_center_position: (num_envs, 3) tensor of adaptive gate center positions.
            gate_passed_current: per-env bool tensor for tracking-tolerance gate passage.
        """
        # Use geometric center of gate opening (z + center_height) so a perfect center pass can approach 0
        gate_center_position = self.task.gate_position.clone()
        gate_center_position[:, 2] = gate_center_position[:, 2] + self.task.gate_center_height
        gate_distance = torch.norm(robot_position - gate_center_position, dim=1)

        # Check if robot has passed gate (crossed Y = 0 plane with proper alignment) - ADAPTIVE
        gate_tracking_width_tolerance = self.task.gate_width * 0.6  # 60% of gate width for tracking
        gate_tracking_min_height = self.task.gate_position[:, 2] + self.task.gate_height * 0.08  # 8% above ground
        gate_tracking_max_height = self.task.gate_position[:, 2] + self.task.gate_height * 0.92  # 92% of gate height

        gate_passed_current = (
            (robot_position[:, 1] > self.task.gate_position[:, 1])
            & (torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0]) < gate_tracking_width_tolerance)
            & (robot_position[:, 2] > gate_tracking_min_height)
            & (robot_position[:, 2] < gate_tracking_max_height)
        )

        # Gate alignment: check if robot is roughly aligned with gate opening - ADAPTIVE
        gate_alignment = torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0]) < gate_tracking_width_tolerance

        # Camera alignment angle in degrees (convert from dot product)
        alignment_angle_deg = torch.acos(torch.clamp(camera_gate_alignment, -1.0, 1.0)) * 180.0 / 3.14159

        # Camera alignment category based on angle
        alignment_category = torch.zeros_like(alignment_angle_deg)
        alignment_category[alignment_angle_deg <= 15] = 5  # Perfect
        alignment_category[(alignment_angle_deg > 15) & (alignment_angle_deg <= 30)] = 4  # Excellent
        alignment_category[(alignment_angle_deg > 30) & (alignment_angle_deg <= 60)] = 3  # Good
        alignment_category[(alignment_angle_deg > 60) & (alignment_angle_deg <= 90)] = 2  # Moderate
        alignment_category[(alignment_angle_deg > 90) & (alignment_angle_deg <= 135)] = 1  # Poor
        alignment_category[alignment_angle_deg > 135] = 0  # Severely misaligned

        self.task.infos["gate/passed"] = gate_passed_current.float()
        self.task.infos["gate/distance"] = gate_distance
        self.task.infos["gate/alignment"] = gate_alignment.float()
        self.task.infos["camera/facing_alignment"] = camera_gate_alignment
        self.task.infos["camera/alignment_angle_deg"] = alignment_angle_deg
        self.task.infos["camera/alignment_category"] = alignment_category

        return gate_center_position, gate_passed_current

    def _update_trajectory_state(
        self, robot_position: torch.Tensor, gate_center_position: torch.Tensor, gate_passed_current: torch.Tensor
    ) -> None:
        """Update per-env episode trajectory tracking: spawn capture, path length, gate crossing."""
        fresh_mask = self.task._episode_fresh
        if torch.any(fresh_mask):
            self.task._ep_spawn_pos[fresh_mask] = robot_position[fresh_mask]
            _gcenter = self.task.gate_position.clone()
            _gcenter[:, 2] = _gcenter[:, 2] + self.task.gate_center_height
            self.task._ep_gate_center_at_spawn[fresh_mask] = _gcenter[fresh_mask]
            self.task._ep_last_pos[fresh_mask] = robot_position[fresh_mask]
            self.task._episode_fresh[fresh_mask] = False

        step_deltas = robot_position - self.task._ep_last_pos
        step_dist = torch.norm(step_deltas, dim=1)
        self.task._ep_path_len += step_dist
        self.task._ep_last_pos = robot_position
        self.task._ep_steps += 1

        step_gate_dist = torch.norm(robot_position - gate_center_position, dim=1)
        self.task._ep_min_gate_dist = torch.minimum(self.task._ep_min_gate_dist, step_gate_dist)

        newly_crossed = (~self.task._ep_gate_crossed) & gate_passed_current
        if torch.any(newly_crossed):
            self.task._ep_gate_crossed[newly_crossed] = True
            self.task._ep_time_to_gate[newly_crossed] = self.task._ep_steps[newly_crossed].to(torch.float32)
            dx_cross = robot_position[newly_crossed, 0] - gate_center_position[newly_crossed, 0]
            dz_cross = robot_position[newly_crossed, 2] - gate_center_position[newly_crossed, 2]
            self.task._ep_center_offset_cross[newly_crossed] = torch.sqrt(dx_cross * dx_cross + dz_cross * dz_cross)
            self.task._ep_height_offset_cross[newly_crossed] = torch.abs(dz_cross)

    def _handle_post_reward_reset(
        self,
        robot_position: torch.Tensor,
        robot_position_before_reset: torch.Tensor,
        gate_center_position: torch.Tensor,
        successes: torch.Tensor,
        target_successes: torch.Tensor,
        reset_envs: torch.Tensor,
    ) -> None:
        """Compute episode-end trajectory metrics, stash infos, and reset completed envs."""
        try:
            env_ids = reset_envs if torch.is_tensor(reset_envs) else torch.tensor(reset_envs, device=self.task.device, dtype=torch.long)
            # Path efficiency = path length / straight-line distance from spawn to gate center at spawn
            denom = torch.norm(self.task._ep_spawn_pos[env_ids] - self.task._ep_gate_center_at_spawn[env_ids], dim=1)
            denom = torch.clamp(denom, min=1e-6)
            # Fallback for rare cases where incremental path stayed ~0 (e.g., immediate reset)
            disp = torch.norm((robot_position[env_ids] - self.task._ep_spawn_pos[env_ids]), dim=1)
            path_len = self.task._ep_path_len[env_ids]
            path_len = torch.where(path_len <= 1e-6, disp, path_len)
            path_eff = torch.full((self.task.num_envs,), float('nan'), device=self.task.device)
            path_eff[env_ids] = (path_len / denom).clamp(max=1000.0)
            # Time to gate in steps (already NaN for non-crossers)
            time_to_gate = self.task._ep_time_to_gate.clone()
            # Min distance to gate center during episode
            min_gate_dist = self.task._ep_min_gate_dist.clone()
            # Offsets at crossing (NaN for non-crossers)
            center_offset = self.task._ep_center_offset_cross.clone()
            height_offset = self.task._ep_height_offset_cross.clone()
            # Last position at episode end (absolute and center-relative distance)
            # Use the snapshot from BEFORE reset to report end-of-episode last pose
            last_pos = robot_position_before_reset[env_ids]
            last_pos_x = last_pos[:, 0]
            last_pos_y = last_pos[:, 1]
            last_pos_z = last_pos[:, 2]
            # Center-relative error (2D XZ) at termination
            dx_last = last_pos_x - gate_center_position[env_ids, 0]
            dz_last = last_pos_z - gate_center_position[env_ids, 2]
            last_center_offset_vals = torch.sqrt(dx_last * dx_last + dz_last * dz_last)
            last_height_offset_vals = torch.abs(dz_last)
            # Debug print: average across resetting envs (NaN-aware)
            pe_avg = torch.nanmean(path_eff[env_ids])
            ttg_avg = torch.nanmean(time_to_gate[env_ids])
            mgd_avg = torch.nanmean(min_gate_dist[env_ids])
            co_avg = torch.nanmean(center_offset[env_ids])
            ho_avg = torch.nanmean(height_offset[env_ids])
            lpx_avg = torch.nanmean(last_pos_x)
            lpy_avg = torch.nanmean(last_pos_y)
            lpz_avg = torch.nanmean(last_pos_z)
            # Also compute episode-end offsets (useful fallback when no crossing occurred)
            lco_avg = torch.nanmean(last_center_offset_vals)
            lho_avg = torch.nanmean(last_height_offset_vals)
            # Report both: overall success rate (gate passage) and target success rate (gate passage AND 10%/10%)
            try:
                # Overall success rate among resetting envs
                overall_success_rate = torch.mean((successes[env_ids] > 0).float())
                # Target success (10% width/height AND gate passage) among resetting envs
                target_success_rate = torch.mean((target_successes[env_ids] > 0).float())
            except (ValueError, TypeError):
                overall_success_rate = torch.tensor(float('nan'), device=self.task.device)
                target_success_rate = torch.tensor(float('nan'), device=self.task.device)
            # Stash per-env episode metrics for worker-side running aggregation
            self.task._stash_per_env_trajectory_metrics(
                env_ids, path_eff, time_to_gate, min_gate_dist, center_offset, height_offset,
                last_pos_x, last_pos_y, last_pos_z, last_center_offset_vals, last_height_offset_vals,
            )
            # Stash the averaged trajectory metrics for logging
            self.task._stash_averaged_trajectory_metrics(
                env_ids, pe_avg, ttg_avg, mgd_avg, co_avg, ho_avg,
                lpx_avg, lpy_avg, lpz_avg, lco_avg, lho_avg,
                overall_success_rate, target_success_rate, time_to_gate,
            )
            # Provide averaged metrics to infos['episode_extra_stats'] so learner can push to W&B as a backup
            self.task._populate_episode_extra_stats()
        except (ValueError, TypeError) as e:
            logger.debug(f"Trajectory metrics computation failed: {e}")
        # Stash infos to return to the learner before we clear them in reset
        self.task._infos_to_return = dict(self.task.infos)
        # Finally, reset environments and mark them fresh for next episode
        self.task.reset_idx(reset_envs)

    def _stash_per_env_trajectory_metrics(
        self,
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
            self.task._last_traj_metrics_per_env = {
                'path_efficiency': path_eff.detach().clone(),
                'time_to_gate_steps': time_to_gate.detach().clone(),
                'min_gate_distance': min_gate_dist.detach().clone(),
                'center_offset_success': center_offset.detach().clone(),
                'height_offset_success': height_offset.detach().clone(),
                'target_success_flag': self.task._ep_target_success_flag.detach().clone(),
                'last_position_x': torch.full((self.task.num_envs,), float('nan'), device=self.task.device),
                'last_position_y': torch.full((self.task.num_envs,), float('nan'), device=self.task.device),
                'last_position_z': torch.full((self.task.num_envs,), float('nan'), device=self.task.device),
                'last_center_offset': torch.full((self.task.num_envs,), float('nan'), device=self.task.device),
                'last_height_offset': torch.full((self.task.num_envs,), float('nan'), device=self.task.device),
                'crossed': self.task._ep_gate_crossed.detach().clone(),
            }
            self.task._last_traj_metrics_per_env['last_position_x'][env_ids] = last_pos_x
            self.task._last_traj_metrics_per_env['last_position_y'][env_ids] = last_pos_y
            self.task._last_traj_metrics_per_env['last_position_z'][env_ids] = last_pos_z
            self.task._last_traj_metrics_per_env['last_center_offset'][env_ids] = last_center_offset_vals
            self.task._last_traj_metrics_per_env['last_height_offset'][env_ids] = last_height_offset_vals
        except (ValueError, TypeError):
            self.task._last_traj_metrics_per_env = None

    def _stash_averaged_trajectory_metrics(
        self,
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
            except Exception:
                co_val = lco_avg
            try:
                ho_val = ho_avg
                if torch.isnan(ho_val):
                    ho_val = lho_avg
            except Exception:
                ho_val = lho_avg
            # Build metrics dict while avoiding undefined time-to-gate when no crossing occurred
            _metrics_avg = {
                'path_efficiency': float(pe_avg.item()),
                'min_gate_distance': float(mgd_avg.item()),
                'center_offset_success': float(co_val.item()) if hasattr(co_val, 'item') else float('nan'),
                'height_offset_success': float(ho_val.item()) if hasattr(ho_val, 'item') else float('nan'),
                # Duplicate keys to match existing dashboards
                'center_offset': float(co_val.item()) if hasattr(co_val, 'item') else float('nan'),
                'height_offset': float(ho_val.item()) if hasattr(ho_val, 'item') else float('nan'),
                'success_rate': float(overall_success_rate.item()) if hasattr(overall_success_rate, 'item') else float('nan'),
                'target_success_rate': float(target_success_rate.item()) if hasattr(target_success_rate, 'item') else float('nan'),
                'last_position_x': float(lpx_avg.item()),
                'last_position_y': float(lpy_avg.item()),
                'last_position_z': float(lpz_avg.item()),
                'last_center_offset': float(lco_avg.item()),
                'last_height_offset': float(lho_avg.item()),
            }
            # Only include time-to-gate (steps/seconds) if any env in this reset batch actually crossed
            try:
                num_crossed = int(torch.isfinite(time_to_gate[env_ids]).sum().item())
            except (ValueError, TypeError):
                num_crossed = 0
            if num_crossed > 0 and not torch.isnan(ttg_avg):
                _metrics_avg['time_to_gate_steps'] = float(ttg_avg.item())
                _metrics_avg['time_to_gate'] = float(ttg_avg.item())
            self.task._last_traj_metrics_avg = _metrics_avg
        except (ValueError, TypeError):
            self.task._last_traj_metrics_avg = None

    def _populate_episode_extra_stats(self) -> None:
        """Populate infos['episode_extra_stats'] with trajectory metrics and camera ablation flags."""
        extra = self.task.infos.get('episode_extra_stats', {})
        if not isinstance(extra, dict):
            extra = {}
        extra.update(self.task._last_traj_metrics_avg or {})
        # Expose per-camera noise/frame-drop overrides to W&B, mirroring prior style
        gtd = getattr(self.task.sim_env, 'global_tensor_dict', {})
        cam_noise_global = bool(gtd.get('camera_randomization/noise_disabled', False))
        cam_fd_global = bool(gtd.get('camera_randomization/frame_dropout_disabled', False))
        drone_noise_dis = bool(gtd.get('camera_randomization/drone_noise_disabled', False)) if 'camera_randomization/drone_noise_disabled' in gtd else cam_noise_global
        static_noise_dis = bool(gtd.get('camera_randomization/static_noise_disabled', False)) if 'camera_randomization/static_noise_disabled' in gtd else cam_noise_global
        drone_fd_dis = bool(gtd.get('camera_randomization/drone_frame_dropout_disabled', False)) if 'camera_randomization/drone_frame_dropout_disabled' in gtd else cam_fd_global
        static_fd_dis = bool(gtd.get('camera_randomization/static_frame_dropout_disabled', False)) if 'camera_randomization/static_frame_dropout_disabled' in gtd else cam_fd_global
        extra['episode_extra_stats/camera_noise_disabled_drone'] = float(drone_noise_dis)
        extra['episode_extra_stats/camera_noise_disabled_static'] = float(static_noise_dis)
        extra['episode_extra_stats/camera_frame_dropout_disabled_drone'] = float(drone_fd_dis)
        extra['episode_extra_stats/camera_frame_dropout_disabled_static'] = float(static_fd_dis)
        self.task.infos['episode_extra_stats'] = extra

    def _process_images_and_finalize(self) -> None:
        """Run image processing, static camera updates, and one-shot final verification."""
        self.task.process_image_observation()
        self.task.process_static_camera_observation()
        self.task.post_image_reward_addition()

        # FINAL VERIFICATION: After all processing is complete
        if not self.task._final_verification_printed:
            self.task._final_verification_printed = True
            logger.warning("FINAL STATIC CAMERA VERIFICATION (AFTER PROCESSING):")

            # Process observations to get final state
            self.task.process_obs_for_task()

            if 'observations' in self.task.task_obs:
                obs_sample = self.task.task_obs["observations"][0]

                static_pos = obs_sample[3:6]
                static_orient = obs_sample[6:9]
                static_vae = obs_sample[86:150]

                logger.warning(f"  Final static pos: {static_pos.cpu().numpy()}")
                logger.warning(f"  Final static orient: {static_orient.cpu().numpy()}")
                logger.warning(f"  Final static VAE: range=[{static_vae.min().item():.3f}, {static_vae.max().item():.3f}]")

                # Check final state
                pos_ok = not torch.allclose(static_pos, torch.zeros_like(static_pos), atol=1e-6)
                orient_ok = not torch.allclose(static_orient, torch.zeros_like(static_orient), atol=1e-6)
                vae_ok = not torch.allclose(static_vae, torch.zeros_like(static_vae), atol=1e-6)

                logger.warning(f"  FINAL RESULTS: pos={pos_ok}, orient={orient_ok}, vae={vae_ok}")

                if pos_ok and orient_ok and vae_ok:
                    logger.warning("SUCCESS: All 150D static camera observations verified!")

                    # CRITICAL: Add verification that observations reach RL training
                    logger.warning("RL TRAINING USAGE VERIFICATION:")
                    logger.warning("  IMPORTANT: This verifies DATA PIPELINE, not RL training usage!")
                    logger.warning("  To verify RL training usage, check:")
                    logger.warning("     1. Neural network receives 150D input (not 128D or other)")
                    logger.warning("     2. Policy network architecture matches observation space")
                    logger.warning("     3. Static camera indices [3:6] and [86:150] affect policy decisions")
                    logger.warning("     4. Ablation test: performance difference with vs without static camera")
                    logger.warning("  Current verification: Environment correctly provides 150D observations")
                    logger.warning("  Next step needed: Verify Sample Factory & neural network usage")
                else:
                    logger.error("Some static camera data still missing after processing!")


