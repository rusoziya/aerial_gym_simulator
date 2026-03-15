from __future__ import annotations

import torch

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import *  # noqa: F401,F403

logger = CustomLogger("navigation_task_gate_reward_tracking")


class RewardTracking:
    def __init__(self, task: object) -> None:
        self.task = task

    def update_episode_reward_tracking(self, obs_dict: dict[str, torch.Tensor], rewards: torch.Tensor, crashes: torch.Tensor) -> None:
        """Update cumulative episode reward tracking for comprehensive debugging."""
        robot_position = obs_dict["robot_position"]

        # Calculate individual reward components (same as in compute_rewards_and_crashes)
        dist = torch.norm(self.task.pos_error_vehicle_frame, dim=1)
        prev_dist = torch.norm(self.task.pos_error_vehicle_frame_prev, dim=1)
        action = obs_dict["robot_actions"].clone()
        prev_action = obs_dict["robot_prev_actions"].clone()

        mult_factor = 1.0 + (0.5) * self.task.curriculum_progress_fraction

        # Position reward
        pos_reward = exponential_reward_function(
            self.task.task_config.reward_parameters["pos_reward_magnitude"],
            self.task.task_config.reward_parameters["pos_reward_exponent"],
            dist,
        )
        self.task.episode_pos_reward += mult_factor * pos_reward

        # Very close reward
        very_close_reward = exponential_reward_function(
            self.task.task_config.reward_parameters["very_close_to_goal_reward_magnitude"],
            self.task.task_config.reward_parameters["very_close_to_goal_reward_exponent"],
            dist,
        )
        self.task.episode_very_close_reward += mult_factor * very_close_reward

        # Getting closer reward
        getting_closer = prev_dist - dist
        getting_closer_reward = torch.where(
            getting_closer > 0,
            self.task.task_config.reward_parameters["getting_closer_reward_multiplier"] * getting_closer,
            2.0 * self.task.task_config.reward_parameters["getting_closer_reward_multiplier"] * getting_closer,
        )
        self.task.episode_getting_closer_reward += mult_factor * getting_closer_reward

        # Gate approach reward
        # Use adaptive gate center (z = bottom + center_height)
        gate_center_position = self.task.gate_position.clone()
        gate_center_position[:, 2] = gate_center_position[:, 2] + self.task.gate_center_height
        gate_distance = torch.norm(robot_position - gate_center_position, dim=1)
        gate_approach_reward = exponential_reward_function(
            self.task.task_config.reward_parameters["gate_approach_reward_magnitude"],
            0.5,
            gate_distance,
        )
        self.task.episode_gate_approach_reward += mult_factor * gate_approach_reward

        # Gate alignment reward
        gate_alignment_reward = torch.zeros_like(gate_distance)
        aligned_mask = torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0]) < 1.5
        gate_alignment_reward[aligned_mask] = self.task.task_config.reward_parameters["gate_alignment_reward_magnitude"]
        self.task.episode_gate_alignment_reward += mult_factor * gate_alignment_reward

        # Camera facing reward (same calculation as in debugging section)
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        drone_to_gate = self.task.gate_position - robot_position
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
        camera_facing_reward[perfect_mask] = self.task.task_config.reward_parameters["camera_facing_reward_magnitude"]
        excellent_mask = (camera_gate_alignment > 0.866) & (camera_gate_alignment <= 0.966)
        camera_facing_reward[excellent_mask] = 0.9 * self.task.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[excellent_mask]
        good_mask = (camera_gate_alignment > 0.5) & (camera_gate_alignment <= 0.866)
        camera_facing_reward[good_mask] = 0.8 * self.task.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[good_mask]
        moderate_mask = (camera_gate_alignment > 0.0) & (camera_gate_alignment <= 0.5)
        camera_facing_reward[moderate_mask] = 0.4 * self.task.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[moderate_mask]
        poor_mask = (camera_gate_alignment > -0.707) & (camera_gate_alignment <= 0.0)
        camera_facing_reward[poor_mask] = 0.2 * self.task.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[poor_mask]
        severe_mask = camera_gate_alignment <= -0.707
        camera_facing_reward[severe_mask] = 2.0 * self.task.task_config.reward_parameters["camera_facing_reward_magnitude"] * camera_gate_alignment[severe_mask]
        # Gate the camera-facing reward: only before first crossing and while approaching (y below gate plane)
        approach_mask = (robot_position[:, 1] < self.task.gate_position[:, 1] - 0.1) & (~self.task.gate_passed)
        camera_facing_reward = camera_facing_reward * approach_mask.float()
        self.task.episode_camera_facing_reward += mult_factor * camera_facing_reward

        # Action penalties - FIXED: Added missing Y-action penalties for 4D action space  
        action_diff = action - prev_action

        x_diff_penalty = exponential_penalty_function(
            self.task.task_config.reward_parameters["x_action_diff_penalty_magnitude"],
            self.task.task_config.reward_parameters["x_action_diff_penalty_exponent"],
            action_diff[:, 0],
        )
        y_diff_penalty = exponential_penalty_function(
            self.task.task_config.reward_parameters["y_action_diff_penalty_magnitude"],
            self.task.task_config.reward_parameters["y_action_diff_penalty_exponent"],
            action_diff[:, 1],
        )
        z_diff_penalty = exponential_penalty_function(
            self.task.task_config.reward_parameters["z_action_diff_penalty_magnitude"],
            self.task.task_config.reward_parameters["z_action_diff_penalty_exponent"],
            action_diff[:, 2],
        )
        yawrate_diff_penalty = exponential_penalty_function(
            self.task.task_config.reward_parameters["yawrate_action_diff_penalty_magnitude"],
            self.task.task_config.reward_parameters["yawrate_action_diff_penalty_exponent"],
            action_diff[:, 3],
        )

        x_absolute_penalty = exponential_penalty_function(
            self.task.task_config.reward_parameters["x_absolute_action_penalty_magnitude"],
            self.task.task_config.reward_parameters["x_absolute_action_penalty_exponent"],
            action[:, 0],
        )
        y_absolute_penalty = exponential_penalty_function(
            self.task.task_config.reward_parameters["y_absolute_action_penalty_magnitude"],
            self.task.task_config.reward_parameters["y_absolute_action_penalty_exponent"],
            action[:, 1],
        )
        z_absolute_penalty = exponential_penalty_function(
            self.task.task_config.reward_parameters["z_absolute_action_penalty_magnitude"],
            self.task.task_config.reward_parameters["z_absolute_action_penalty_exponent"],
            action[:, 2],
        )
        yawrate_absolute_penalty = exponential_penalty_function(
            self.task.task_config.reward_parameters["yawrate_absolute_action_penalty_magnitude"],
            self.task.task_config.reward_parameters["yawrate_absolute_action_penalty_exponent"],
            action[:, 3],
        )

        action_diff_penalty = x_diff_penalty + y_diff_penalty + z_diff_penalty + yawrate_diff_penalty
        absolute_action_penalty = x_absolute_penalty + y_absolute_penalty + z_absolute_penalty + yawrate_absolute_penalty
        total_action_penalty = action_diff_penalty + absolute_action_penalty
        self.task.episode_action_penalty += total_action_penalty

        # Track collision penalties
        collision_mask = crashes > 0
        collision_penalty = torch.where(
            collision_mask,
            self.task.task_config.reward_parameters["collision_penalty"],
            torch.zeros_like(crashes, dtype=torch.float32),
        )
        self.task.episode_collision_penalty += collision_penalty

        # Track gate passage rewards (check if any gate passages occurred this step) - ADAPTIVE
        # Use the same logic as main reward system with adaptive dimensions
        tracking_width_tolerance = self.task.gate_width * 0.6  # 60% of gate width
        tracking_min_height = self.task.gate_position[:, 2] + self.task.gate_height * 0.08  # 8% above ground
        tracking_max_height = self.task.gate_position[:, 2] + self.task.gate_height * 0.92  # 92% of gate height

        gate_passed_this_step = (
            (robot_position[:, 1] > self.task.gate_position[:, 1]) &
            (torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0]) < tracking_width_tolerance) &
            (robot_position[:, 2] > tracking_min_height) & (robot_position[:, 2] < tracking_max_height) &
            (~self.task.gate_passed)  # Haven't passed before
        )

        # Center passage detection with adaptive dimensions (like main system)
        x_distance_from_center = torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0])
        z_distance_from_center = torch.abs(robot_position[:, 2] - (self.task.gate_position[:, 2] + self.task.gate_center_height))

        # Adaptive center thresholds
        center_x_threshold = self.task.gate_width * 0.2  # 20% of gate width for center alignment
        center_z_threshold = self.task.gate_height * 0.125  # 12.5% of gate height for center alignment
        center_aligned_mask = (x_distance_from_center < center_x_threshold) & (z_distance_from_center < center_z_threshold)

        # Basic gate passage reward
        gate_passage_reward = torch.where(
            gate_passed_this_step,
            mult_factor * self.task.task_config.reward_parameters["gate_passage_reward_magnitude"],
            torch.zeros_like(gate_passed_this_step, dtype=torch.float32),
        )

        # Gate center passage bonus (only for centered passages)
        gate_center_passage_bonus = torch.where(
            gate_passed_this_step & center_aligned_mask,
            mult_factor * self.task.task_config.reward_parameters["gate_center_passage_bonus_magnitude"],
            torch.zeros_like(gate_passed_this_step, dtype=torch.float32),
        )

        self.task.gate_passed = self.task.gate_passed | gate_passed_this_step

        # Track total gate passage rewards (basic + center bonus)
        total_gate_rewards = gate_passage_reward + gate_center_passage_bonus
        self.task.episode_gate_passage_reward += total_gate_rewards

        # Boundary violation penalty (episode tracking mirror of TorchScript path)
        y_margin = 0.2
        behind_gate_mask = (robot_position[:, 1] > (self.task.gate_position[:, 1] + y_margin))
        full_width_tol = self.task.gate_width * 0.5
        full_min_h = self.task.gate_position[:, 2] + 0.0 * self.task.gate_height
        full_max_h = self.task.gate_position[:, 2] + 1.0 * self.task.gate_height
        within_passage_window = (
            (torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0]) < full_width_tol)
            & (robot_position[:, 2] > full_min_h)
            & (robot_position[:, 2] < full_max_h)
        )
        misaligned_cross_mask = behind_gate_mask & (~within_passage_window) & (~self.task.gate_passed) & (~gate_passed_this_step)
        boundary_violation_penalty = torch.zeros_like(gate_distance)
        rising_mask = misaligned_cross_mask & (~self.task._bv_flag_episode)
        boundary_violation_penalty[rising_mask] = -50.0
        self.task._bv_flag_episode = self.task._bv_flag_episode | misaligned_cross_mask
        if torch.any(rising_mask):
            self.task.terminations[rising_mask] = 1
        self.task.episode_boundary_violation_penalty += boundary_violation_penalty

        # Track image rewards (from post_image_reward_addition)
        if self.task.min_pixel_dist is not None:
            mag = self.task.task_config.reward_parameters.get("image_penalty_magnitude", 4.0)
            expo = self.task.task_config.reward_parameters.get("image_penalty_exponent", 1.0)
            image_rewards = -exponential_reward_function(
                float(mag), float(expo), self.task.min_pixel_dist[~self.task.terminations]
            )
            # Only add for non-terminated environments
            non_terminated_mask = ~self.task.terminations
            if torch.sum(non_terminated_mask) > 0:
                self.task.episode_image_reward[non_terminated_mask] += image_rewards

        # Increment episode length tracking
        self.task.episode_lengths += 1


    def reset_episode_reward_tracking(self, env_ids: torch.Tensor) -> None:
        """Reset episode reward tracking for specified environments when episodes end."""
        if len(env_ids) == 0:
            return

        # Store completed episode data for averaging
        for env_id in env_ids:
            if self.task.episode_lengths[env_id] > 0:  # Valid episode
                episode_data = {
                    'total_reward': (
                        self.task.episode_pos_reward[env_id] + 
                        self.task.episode_very_close_reward[env_id] + 
                        self.task.episode_getting_closer_reward[env_id] + 
                        self.task.episode_gate_approach_reward[env_id] + 
                        self.task.episode_gate_alignment_reward[env_id] + 
                        self.task.episode_camera_facing_reward[env_id] + 
                        self.task.episode_action_penalty[env_id] + 
                        self.task.episode_time_penalty[env_id] + 
                        self.task.episode_timeout_penalty[env_id] + 
                        self.task.episode_boundary_violation_penalty[env_id] + 
                        self.task.episode_gate_passage_reward[env_id] + 
                        self.task.episode_collision_penalty[env_id] + 
                        self.task.episode_image_reward[env_id]
                    ).item(),
                    'pos_reward': self.task.episode_pos_reward[env_id].item(),
                    'very_close_reward': self.task.episode_very_close_reward[env_id].item(),
                    'getting_closer_reward': self.task.episode_getting_closer_reward[env_id].item(),
                    'gate_approach_reward': self.task.episode_gate_approach_reward[env_id].item(),
                    'gate_alignment_reward': self.task.episode_gate_alignment_reward[env_id].item(),
                    'camera_facing_reward': self.task.episode_camera_facing_reward[env_id].item(),
                    'action_penalty': self.task.episode_action_penalty[env_id].item(),
                    'boundary_violation_penalty': self.task.episode_boundary_violation_penalty[env_id].item(),
                    'time_penalty': self.task.episode_time_penalty[env_id].item(),
                    'timeout_penalty': self.task.episode_timeout_penalty[env_id].item(),
                    'gate_passage_reward': self.task.episode_gate_passage_reward[env_id].item(),  # Now includes both basic + center bonus
                    'collision_penalty': self.task.episode_collision_penalty[env_id].item(),
                    'image_reward': self.task.episode_image_reward[env_id].item(),
                    'episode_length': self.task.episode_lengths[env_id].item(),
                }
                self.task.completed_episodes.append(episode_data)

                # Keep only last N episodes
                if len(self.task.completed_episodes) > self.task.max_stored_episodes:
                    self.task.completed_episodes.pop(0)

        # Reset trackers for completed episodes
        self.task.episode_rewards.reset_envs(env_ids)


    def _log_comprehensive_reward_debug(
        self,
        obs_dict: dict[str, torch.Tensor],
        rewards: torch.Tensor,
        crashes: torch.Tensor,
        boundary_violation_one_shot_mask: torch.Tensor,
        camera_gate_alignment: torch.Tensor,
    ) -> None:
        """Log reward summary and outliers (gated by config flag and step interval)."""
        if (
            self.task.num_task_steps % 200 == 0
            and bool(self.task.task_config.enable_comprehensive_reward_debug)
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
        robot_position = obs_dict["robot_position"]
        dist = torch.norm(self.task.pos_error_vehicle_frame, dim=1)

        mult_factor = float(self.task._curriculum_multiplier_factor)
        avg_reward = torch.mean(rewards).item()
        avg_dist = torch.mean(dist).item()
        num_crashes = int(torch.sum(crashes).item())
        avg_ep_len = torch.mean(self.task.episode_lengths.float()).item()

        logger.info(
            f"[RewardDebug] step={self.task.num_task_steps} "
            f"avg_reward={avg_reward:.3f} avg_dist={avg_dist:.2f}m "
            f"crashes={num_crashes} mult={mult_factor:.2f} avg_ep_len={avg_ep_len:.0f}"
        )

        if len(self.task.completed_episodes) > 0:
            avg_ep = {}
            for key in self.task.completed_episodes[0]:
                avg_ep[key] = sum(
                    ep[key] for ep in self.task.completed_episodes
                ) / len(self.task.completed_episodes)
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



