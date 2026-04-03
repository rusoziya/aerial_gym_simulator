from __future__ import annotations

import torch

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import *  # noqa: F401,F403

logger = CustomLogger("navigation_task_gate_reward_tracking")


class RewardTracking:
    def __init__(self, task: object) -> None:
        self.task = task

    def update_episode_reward_tracking(
        self, obs_dict: dict[str, torch.Tensor], rewards: torch.Tensor, crashes: torch.Tensor
    ) -> None:
        """Update cumulative episode reward tracking for comprehensive debugging."""
        robot_position = obs_dict["robot_position"]
        dist = torch.norm(self.task.pos_error_vehicle_frame, dim=1)
        prev_dist = torch.norm(self.task.pos_error_vehicle_frame_prev, dim=1)
        action = obs_dict["robot_actions"].clone()
        prev_action = obs_dict["robot_prev_actions"].clone()

        mult_factor = 1.0 + (0.5) * self.task.curriculum_progress_fraction

        self._track_position_rewards(dist, prev_dist, mult_factor)
        self._track_gate_rewards(robot_position, dist, mult_factor)
        self._track_camera_facing_reward(obs_dict, robot_position, mult_factor)
        self._track_action_penalties(action, prev_action)
        self._track_collision_penalty(crashes)
        self._track_gate_passage(robot_position, mult_factor)
        self._track_boundary_violation(robot_position)
        self._track_image_reward()
        self.task.episode_lengths += 1

    def _track_position_rewards(
        self, dist: torch.Tensor, prev_dist: torch.Tensor, mult_factor: float
    ) -> None:
        rp = self.task.task_config.reward_parameters
        pos_reward = exponential_reward_function(
            rp["pos_reward_magnitude"], rp["pos_reward_exponent"], dist
        )
        self.task.episode_pos_reward += mult_factor * pos_reward

        very_close_reward = exponential_reward_function(
            rp["very_close_to_goal_reward_magnitude"],
            rp["very_close_to_goal_reward_exponent"],
            dist,
        )
        self.task.episode_very_close_reward += mult_factor * very_close_reward

        getting_closer = prev_dist - dist
        getting_closer_reward = torch.where(
            getting_closer > 0,
            rp["getting_closer_reward_multiplier"] * getting_closer,
            2.0 * rp["getting_closer_reward_multiplier"] * getting_closer,
        )
        self.task.episode_getting_closer_reward += mult_factor * getting_closer_reward

    def _track_gate_rewards(
        self, robot_position: torch.Tensor, dist: torch.Tensor, mult_factor: float
    ) -> None:
        rp = self.task.task_config.reward_parameters
        gate_center_position = self.task.gate_position.clone()
        gate_center_position[:, 2] = gate_center_position[:, 2] + self.task.gate_center_height
        gate_distance = torch.norm(robot_position - gate_center_position, dim=1)

        gate_approach_reward = exponential_reward_function(
            rp["gate_approach_reward_magnitude"], 0.5, gate_distance
        )
        self.task.episode_gate_approach_reward += mult_factor * gate_approach_reward

        gate_alignment_reward = torch.zeros_like(gate_distance)
        aligned_mask = torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0]) < 1.5
        gate_alignment_reward[aligned_mask] = rp["gate_alignment_reward_magnitude"]
        self.task.episode_gate_alignment_reward += mult_factor * gate_alignment_reward

    def _track_camera_facing_reward(
        self,
        obs_dict: dict[str, torch.Tensor],
        robot_position: torch.Tensor,
        mult_factor: float,
    ) -> None:
        rp = self.task.task_config.reward_parameters
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]

        drone_to_gate = self.task.gate_position - robot_position
        drone_to_gate_normalized = drone_to_gate / (
            torch.norm(drone_to_gate, dim=1, keepdim=True) + 1e-8
        )

        qw = robot_vehicle_orientation[:, 3]
        qx = robot_vehicle_orientation[:, 0]
        qy = robot_vehicle_orientation[:, 1]
        qz = robot_vehicle_orientation[:, 2]
        forward_x = 1.0 - 2.0 * (qy * qy + qz * qz)
        forward_y = 2.0 * (qx * qy + qw * qz)
        forward_z = 2.0 * (qx * qz - qw * qy)
        drone_forward = torch.stack([forward_x, forward_y, forward_z], dim=1)
        drone_forward_normalized = drone_forward / (
            torch.norm(drone_forward, dim=1, keepdim=True) + 1e-8
        )

        alignment = torch.clamp(
            torch.sum(drone_forward_normalized * drone_to_gate_normalized, dim=1), -1.0, 1.0
        )
        camera_facing_reward = self._compute_tiered_camera_reward(alignment, rp)

        approach_mask = (robot_position[:, 1] < self.task.gate_position[:, 1] - 0.1) & (
            ~self.task.gate_passed
        )
        camera_facing_reward = camera_facing_reward * approach_mask.float()
        self.task.episode_camera_facing_reward += mult_factor * camera_facing_reward

    def _compute_tiered_camera_reward(
        self, alignment: torch.Tensor, rp: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        mag = rp["camera_facing_reward_magnitude"]
        reward = torch.zeros_like(alignment)
        reward[alignment > 0.966] = mag
        excellent = (alignment > 0.866) & (alignment <= 0.966)
        reward[excellent] = 0.9 * mag * alignment[excellent]
        good = (alignment > 0.5) & (alignment <= 0.866)
        reward[good] = 0.8 * mag * alignment[good]
        moderate = (alignment > 0.0) & (alignment <= 0.5)
        reward[moderate] = 0.4 * mag * alignment[moderate]
        poor = (alignment > -0.707) & (alignment <= 0.0)
        reward[poor] = 0.2 * mag * alignment[poor]
        severe = alignment <= -0.707
        reward[severe] = 2.0 * mag * alignment[severe]
        return reward

    def _track_action_penalties(self, action: torch.Tensor, prev_action: torch.Tensor) -> None:
        rp = self.task.task_config.reward_parameters
        action_diff = action - prev_action

        diff_penalty = sum(
            exponential_penalty_function(
                rp[f"{axis}_action_diff_penalty_magnitude"],
                rp[f"{axis}_action_diff_penalty_exponent"],
                action_diff[:, i],
            )
            for i, axis in enumerate(["x", "y", "z", "yawrate"])
        )
        abs_penalty = sum(
            exponential_penalty_function(
                rp[f"{axis}_absolute_action_penalty_magnitude"],
                rp[f"{axis}_absolute_action_penalty_exponent"],
                action[:, i],
            )
            for i, axis in enumerate(["x", "y", "z", "yawrate"])
        )
        self.task.episode_action_penalty += diff_penalty + abs_penalty

    def _track_collision_penalty(self, crashes: torch.Tensor) -> None:
        collision_mask = crashes > 0
        collision_penalty = torch.where(
            collision_mask,
            self.task.task_config.reward_parameters["collision_penalty"],
            torch.zeros_like(crashes, dtype=torch.float32),
        )
        self.task.episode_collision_penalty += collision_penalty

    def _track_gate_passage(self, robot_position: torch.Tensor, mult_factor: float) -> None:
        rp = self.task.task_config.reward_parameters
        tracking_width_tolerance = self.task.gate_width * 0.6
        tracking_min_height = self.task.gate_position[:, 2] + self.task.gate_height * 0.08
        tracking_max_height = self.task.gate_position[:, 2] + self.task.gate_height * 0.92

        gate_passed_this_step = (
            (robot_position[:, 1] > self.task.gate_position[:, 1])
            & (
                torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0])
                < tracking_width_tolerance
            )
            & (robot_position[:, 2] > tracking_min_height)
            & (robot_position[:, 2] < tracking_max_height)
            & (~self.task.gate_passed)
        )

        x_from_center = torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0])
        z_from_center = torch.abs(
            robot_position[:, 2] - (self.task.gate_position[:, 2] + self.task.gate_center_height)
        )
        center_aligned_mask = (x_from_center < self.task.gate_width * 0.2) & (
            z_from_center < self.task.gate_height * 0.125
        )

        gate_passage_reward = torch.where(
            gate_passed_this_step,
            mult_factor * rp["gate_passage_reward_magnitude"],
            torch.zeros_like(gate_passed_this_step, dtype=torch.float32),
        )
        gate_center_bonus = torch.where(
            gate_passed_this_step & center_aligned_mask,
            mult_factor * rp["gate_center_passage_bonus_magnitude"],
            torch.zeros_like(gate_passed_this_step, dtype=torch.float32),
        )

        self.task.gate_passed = self.task.gate_passed | gate_passed_this_step
        self.task.episode_gate_passage_reward += gate_passage_reward + gate_center_bonus

    def _track_boundary_violation(self, robot_position: torch.Tensor) -> None:
        y_margin = 0.2
        behind_gate_mask = robot_position[:, 1] > (self.task.gate_position[:, 1] + y_margin)
        full_width_tol = self.task.gate_width * 0.5
        full_min_h = self.task.gate_position[:, 2] + 0.0 * self.task.gate_height
        full_max_h = self.task.gate_position[:, 2] + 1.0 * self.task.gate_height
        within_passage_window = (
            (torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0]) < full_width_tol)
            & (robot_position[:, 2] > full_min_h)
            & (robot_position[:, 2] < full_max_h)
        )
        misaligned_cross_mask = (
            behind_gate_mask & (~within_passage_window) & (~self.task.gate_passed)
        )

        gate_distance = torch.norm(robot_position - self.task.gate_position, dim=1)
        boundary_violation_penalty = torch.zeros_like(gate_distance)
        rising_mask = misaligned_cross_mask & (~self.task._bv_flag_episode)
        boundary_violation_penalty[rising_mask] = -50.0
        self.task._bv_flag_episode = self.task._bv_flag_episode | misaligned_cross_mask
        if torch.any(rising_mask):
            self.task.terminations[rising_mask] = 1
        self.task.episode_boundary_violation_penalty += boundary_violation_penalty

    def _track_image_reward(self) -> None:
        if self.task.min_pixel_dist is None:
            return
        mag = self.task.task_config.reward_parameters.get("image_penalty_magnitude", 4.0)
        expo = self.task.task_config.reward_parameters.get("image_penalty_exponent", 1.0)
        non_terminated_mask = ~self.task.terminations
        if torch.sum(non_terminated_mask) > 0:
            image_rewards = -exponential_reward_function(
                float(mag), float(expo), self.task.min_pixel_dist[non_terminated_mask]
            )
            self.task.episode_image_reward[non_terminated_mask] += image_rewards

    def reset_episode_reward_tracking(self, env_ids: torch.Tensor) -> None:
        """Reset episode reward tracking for specified environments when episodes end."""
        if len(env_ids) == 0:
            return

        for env_id in env_ids:
            if self.task.episode_lengths[env_id] > 0:
                episode_data = self._collect_episode_data(env_id)
                self.task.completed_episodes.append(episode_data)
                if len(self.task.completed_episodes) > self.task.max_stored_episodes:
                    self.task.completed_episodes.pop(0)

        self.task.episode_rewards.reset_envs(env_ids)

    def _collect_episode_data(self, env_id: int) -> dict[str, float]:
        t = self.task
        return {
            "total_reward": (
                t.episode_pos_reward[env_id]
                + t.episode_very_close_reward[env_id]
                + t.episode_getting_closer_reward[env_id]
                + t.episode_gate_approach_reward[env_id]
                + t.episode_gate_alignment_reward[env_id]
                + t.episode_camera_facing_reward[env_id]
                + t.episode_action_penalty[env_id]
                + t.episode_time_penalty[env_id]
                + t.episode_timeout_penalty[env_id]
                + t.episode_boundary_violation_penalty[env_id]
                + t.episode_gate_passage_reward[env_id]
                + t.episode_collision_penalty[env_id]
                + t.episode_image_reward[env_id]
            ).item(),
            "pos_reward": t.episode_pos_reward[env_id].item(),
            "very_close_reward": t.episode_very_close_reward[env_id].item(),
            "getting_closer_reward": t.episode_getting_closer_reward[env_id].item(),
            "gate_approach_reward": t.episode_gate_approach_reward[env_id].item(),
            "gate_alignment_reward": t.episode_gate_alignment_reward[env_id].item(),
            "camera_facing_reward": t.episode_camera_facing_reward[env_id].item(),
            "action_penalty": t.episode_action_penalty[env_id].item(),
            "boundary_violation_penalty": t.episode_boundary_violation_penalty[env_id].item(),
            "time_penalty": t.episode_time_penalty[env_id].item(),
            "timeout_penalty": t.episode_timeout_penalty[env_id].item(),
            "gate_passage_reward": t.episode_gate_passage_reward[env_id].item(),
            "collision_penalty": t.episode_collision_penalty[env_id].item(),
            "image_reward": t.episode_image_reward[env_id].item(),
            "episode_length": t.episode_lengths[env_id].item(),
        }

    def _log_comprehensive_reward_debug(
        self,
        obs_dict: dict[str, torch.Tensor],
        rewards: torch.Tensor,
        crashes: torch.Tensor,
        boundary_violation_one_shot_mask: torch.Tensor,
        camera_gate_alignment: torch.Tensor,
    ) -> None:
        """Log reward summary and outliers (gated by config flag and step interval)."""
        if self.task.num_task_steps % 200 == 0 and bool(
            self.task.task_config.enable_comprehensive_reward_debug
        ):
            self._log_reward_summary(obs_dict, rewards, crashes)

        self._log_reward_outliers(obs_dict, rewards, crashes, boundary_violation_one_shot_mask)

    def _log_reward_summary(
        self,
        obs_dict: dict[str, torch.Tensor],
        rewards: torch.Tensor,
        crashes: torch.Tensor,
    ) -> None:
        """Log averaged reward components and episode-level breakdown."""
        dist = torch.norm(self.task.pos_error_vehicle_frame, dim=1)
        mult_factor = float(self.task._curriculum_multiplier_factor)
        logger.info(
            f"[RewardDebug] step={self.task.num_task_steps} "
            f"avg_reward={torch.mean(rewards).item():.3f} avg_dist={torch.mean(dist).item():.2f}m "
            f"crashes={int(torch.sum(crashes).item())} mult={mult_factor:.2f} "
            f"avg_ep_len={torch.mean(self.task.episode_lengths.float()).item():.0f}"
        )

        if len(self.task.completed_episodes) > 0:
            avg_ep: dict[str, float] = {}
            for key in self.task.completed_episodes[0]:
                avg_ep[key] = sum(ep[key] for ep in self.task.completed_episodes) / len(
                    self.task.completed_episodes
                )
            logger.info(
                f"[EpisodeAvg] total={avg_ep['total_reward']:.1f} "
                f"pos={avg_ep['pos_reward']:.1f} closer={avg_ep['getting_closer_reward']:.1f} "
                f"gate_approach={avg_ep['gate_approach_reward']:.1f} "
                f"gate_align={avg_ep['gate_alignment_reward']:.1f} "
                f"cam_face={avg_ep['camera_facing_reward']:.1f} "
                f"action_pen={avg_ep['action_penalty']:.1f} "
                f"gate_pass={avg_ep['gate_passage_reward']:.1f} "
                f"collision={avg_ep['collision_penalty']:.1f} "
                f"ep_len={avg_ep['episode_length']:.0f}"
            )

    def _log_reward_outliers(
        self,
        obs_dict: dict[str, torch.Tensor],
        rewards: torch.Tensor,
        crashes: torch.Tensor,
        boundary_violation_one_shot_mask: torch.Tensor,
    ) -> None:
        """Log reward outliers that exceed the configured threshold."""
        thr = float(self.task.task_config.reward_outlier_threshold)
        if not torch.any(rewards < thr):
            return
        bad_ids = torch.nonzero(rewards < thr, as_tuple=False).squeeze(-1)
        if bad_ids.numel() == 0:
            return
        limit = int(self.task.task_config.reward_outlier_log_limit_per_step)
        sample = bad_ids[:limit]
        dist_sample = torch.norm(self.task.pos_error_vehicle_frame[sample], dim=1)
        y_sample = obs_dict["robot_position"][sample, 1]
        bv_sample = boundary_violation_one_shot_mask[sample].tolist()
        logger.warning(
            f"[RewardOutlier] envs={sample.tolist()} rewards={rewards[sample].tolist()} "
            f"crashes={crashes[sample].tolist()}"
        )
        logger.warning(
            f"[RewardOutlier] dist={dist_sample.tolist()} y={y_sample.tolist()} "
            f"boundary_violation={bv_sample}"
        )
