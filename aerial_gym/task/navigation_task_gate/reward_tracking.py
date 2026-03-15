from __future__ import annotations

import os
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
        # One-shot penalty per episode: apply only on rising edge
        if not True:
            self.task._bv_flag_episode = torch.zeros(self.task.num_envs, dtype=torch.bool, device=self.task.device)
        rising_mask = misaligned_cross_mask & (~self.task._bv_flag_episode)
        boundary_violation_penalty[rising_mask] = -50.0
        # Update flag and force termination for violating envs
        self.task._bv_flag_episode = self.task._bv_flag_episode | misaligned_cross_mask
        if torch.any(rising_mask):
            # Use terminations for boundary violations (true MDP failure)
            self.task.terminations[rising_mask] = 1
            # env0 debug already prints above on rising edge
        self.task.episode_boundary_violation_penalty += boundary_violation_penalty
        # Print on rising edge for all envs that violated this step
        rising_envs = torch.nonzero(rising_mask, as_tuple=False).squeeze(-1)
        if rising_envs.numel() > 0:
            for eid in rising_envs.tolist():
                rx = float(robot_position[eid, 0].item())
                ry = float(robot_position[eid, 1].item())
                rz = float(robot_position[eid, 2].item())
                gate_x = float(self.task.gate_position[eid, 0].item())
                gate_y = float(self.task.gate_position[eid, 1].item())
                x_off = abs(rx - gate_x)
                gw = float(self.task.gate_width[eid].item() if hasattr(self.task.gate_width, 'shape') else self.task.gate_width)
                gh = float(self.task.gate_height[eid].item() if hasattr(self.task.gate_height, 'shape') else self.task.gate_height)
                tol = float(full_width_tol[eid].item() if hasattr(full_width_tol, 'shape') else full_width_tol)
                zmin = float(full_min_h[eid].item() if hasattr(full_min_h, 'shape') else full_min_h)
                zmax = float(full_max_h[eid].item() if hasattr(full_max_h, 'shape') else full_max_h)
                logger.warning(
                    f"[Boundary] Env{eid} VIOLATION at step {self.task.num_task_steps}: pos=({rx:.3f},{ry:.3f},{rz:.3f}), "
                    f"gate_y={gate_y:.3f}, x_off={x_off:.3f} (tol={tol:.3f}), z_window=({zmin:.3f},{zmax:.3f}), "
                    f"gate_size=(w={gw:.3f}, h={gh:.3f})"
                )

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
        """Recalculate and log all reward components (every 200 steps, gated by config flag)."""
        # COMPREHENSIVE REWARD DEBUGGING: Print ALL reward components every 200 steps
        # Disabled by default via config flag `enable_comprehensive_reward_debug`
        if (
            True
            and self.task.num_task_steps % 200 == 0
            and bool(self.task.task_config.enable_comprehensive_reward_debug)
        ):
            # Recalculate components for debugging (without JIT optimization)
            dist = torch.norm(self.task.pos_error_vehicle_frame, dim=1)
            prev_dist = torch.norm(self.task.pos_error_vehicle_frame_prev, dim=1)
            action = obs_dict["robot_actions"]
            prev_action = obs_dict["robot_prev_actions"]
            robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]

            # Individual reward components (average across environments)
            pos_reward = exponential_reward_function(
                self.task.task_config.reward_parameters["pos_reward_magnitude"],
                self.task.task_config.reward_parameters["pos_reward_exponent"],
                dist,
            )

            very_close_reward = exponential_reward_function(
                self.task.task_config.reward_parameters["very_close_to_goal_reward_magnitude"],
                self.task.task_config.reward_parameters["very_close_to_goal_reward_exponent"],
                dist,
            )

            getting_closer = prev_dist - dist
            getting_closer_reward = torch.where(
                getting_closer > 0,
                self.task.task_config.reward_parameters["getting_closer_reward_multiplier"] * getting_closer,
                2.0 * self.task.task_config.reward_parameters["getting_closer_reward_multiplier"] * getting_closer,
            )

            # Use adaptive gate center (z = bottom + center_height)
            gate_center_position = self.task.gate_position.clone()
            gate_center_position[:, 2] = gate_center_position[:, 2] + self.task.gate_center_height
            gate_distance = torch.norm(robot_position - gate_center_position, dim=1)
            gate_approach_reward = exponential_reward_function(
                self.task.task_config.reward_parameters["gate_approach_reward_magnitude"],
                0.5,
                gate_distance,
            )

            # Gate alignment (finer piecewise by lateral offset from gate center)
            gate_alignment_reward = torch.zeros_like(gate_distance)
            dx = torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0])
            # Thresholds proportional to current gate width (denser piecewise bins)
            t00 = self.task.gate_width * 0.01
            t0  = self.task.gate_width * 0.02
            t1  = self.task.gate_width * 0.04
            t2  = self.task.gate_width * 0.06
            t3  = self.task.gate_width * 0.08
            t4  = self.task.gate_width * 0.10
            t5  = self.task.gate_width * 0.12
            t6  = self.task.gate_width * 0.15
            t7  = self.task.gate_width * 0.20
            t8  = self.task.gate_width * 0.25
            t9  = self.task.gate_width * 0.30
            t10 = self.task.gate_width * 0.40
            t11 = self.task.gate_width * 0.50
            mag = self.task.task_config.reward_parameters["gate_alignment_reward_magnitude"]
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
            drone_to_gate = self.task.gate_position - robot_position
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

            # Action penalties - FIXED: Added missing Y-action penalties for 4D action space
            action_diff = action - prev_action

            # ENHANCED ACTION DEBUG: Deep investigation of action tracking system
            if self.task.num_task_steps % 200 == 0:
                avg_action_diff = torch.mean(torch.abs(action_diff), dim=0)
                max_action_diff = torch.max(torch.abs(action_diff), dim=0)[0]

                # Show actual action values to understand the pattern
                avg_current = torch.mean(action, dim=0)
                avg_previous = torch.mean(prev_action, dim=0)

                # Check if all actions are identical across environments
                action_std = torch.std(action, dim=0)
                prev_action_std = torch.std(prev_action, dim=0)

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

            # Calculate averages for debugging
            mult_factor = 1.0 + (0.5) * self.task.curriculum_progress_fraction
            avg_total_reward = torch.mean(rewards).item()
            # Use the effective multiplier factor computed earlier in this step
            try:
                mult_factor = float(self.task._curriculum_multiplier_factor)
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
            behind_gate_mask = (robot_position[:, 1] > (self.task.gate_position[:, 1] + y_margin))
            gate_passage_width_tolerance = self.task.gate_width * 0.6
            gate_min_height = self.task.gate_position[:, 2] + self.task.gate_height * 0.1
            gate_max_height = self.task.gate_position[:, 2] + self.task.gate_height * 0.9
            within_passage_window = (
                (torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0]) < gate_passage_width_tolerance)
                & (robot_position[:, 2] > gate_min_height)
                & (robot_position[:, 2] < gate_max_height)
            )
            misaligned_cross_mask = behind_gate_mask & (~within_passage_window) & (~self.task.gate_passed)
            boundary_violation_penalty = torch.zeros_like(gate_distance)
            boundary_violation_penalty[misaligned_cross_mask] = -50.0
            avg_boundary_penalty = torch.mean(boundary_violation_penalty).item()
            avg_distance = torch.mean(dist).item()
            avg_gate_distance = torch.mean(gate_distance).item()
            avg_camera_alignment = torch.mean(camera_gate_alignment).item()
            # Static FOV (recompute shaped-average for logging)
            try:
                fov_mag = float(self.task.task_config.reward_parameters.get("static_fov_visibility_reward_magnitude", 0.0))
            except (ValueError, TypeError):
                fov_mag = 0.0
            avg_static_fov_reward = 0.0
            if fov_mag != 0.0:
                try:
                    parent = self.task.sim_env
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
                    gate_center_z = self.task.gate_center_height
                else:
                    gate_center_z = torch.full((self.task.num_envs,), 1.5, device=self.task.device)
                cam_pos = torch.stack([
                    torch.zeros(self.task.num_envs, device=self.task.device),
                    torch.full((self.task.num_envs,), base_y, device=self.task.device),
                    gate_center_z
                ], dim=1)
                target = torch.stack([
                    torch.zeros(self.task.num_envs, device=self.task.device),
                    torch.zeros(self.task.num_envs, device=self.task.device),
                    gate_center_z
                ], dim=1)
                fwd = target - cam_pos
                fwd = fwd / (torch.norm(fwd, dim=1, keepdim=True) + 1e-8)
                up_world = torch.tensor([0.0, 0.0, 1.0], device=self.task.device).view(1, 3).expand_as(fwd)
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
                    fov_alpha = float(self.task.task_config.reward_parameters.get("static_fov_visibility_exponent", 2.0))
                except (ValueError, TypeError):
                    fov_alpha = 2.0
                fov_score = torch.pow(torch.clamp(1.0 - m_norm, min=0.0), fov_alpha)
                avg_static_fov_reward = float(torch.mean(fov_mag * fov_score).item())

            logger.warning("="*80)
            logger.warning(f"🔍 COMPREHENSIVE REWARD BREAKDOWN (Step {self.task.num_task_steps}):")
            logger.warning(f"  📊 TOTAL REWARD:           {avg_total_reward:.3f}")
            # Print VAE latent statistics alongside reward breakdown for clear visibility
            if isinstance(self.task.task_obs, dict) and 'observations' in self.task.task_obs:
                obs_all = self.task.task_obs['observations']
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
                avg_time_pen = float(torch.mean(self.task.episode_time_penalty).item())
            except (ValueError, TypeError):
                avg_time_pen = 0.0
            try:
                avg_timeout_pen = float(torch.mean(self.task.episode_timeout_penalty).item())
            except (ValueError, TypeError):
                avg_timeout_pen = 0.0
            logger.warning(f"  ⏱️ Time Penalty (avg):     {avg_time_pen:.3f}")
            logger.warning(f"  ⌛ Timeout Penalty (avg):  {avg_timeout_pen:.3f}")
            logger.warning(f"  ⛔ Boundary Violation:     {avg_boundary_penalty:.3f}")
            logger.warning(f"  ⚡ Multiplier Factor:      {mult_factor:.3f}")

            # Check for any gate passages - ADAPTIVE to gate dimensions
            curriculum_width_tolerance = self.task.gate_width * 0.6  # 60% of gate width
            curriculum_min_height = self.task.gate_position[:, 2] + self.task.gate_height * 0.08  # 8% above ground
            curriculum_max_height = self.task.gate_position[:, 2] + self.task.gate_height * 0.92  # 92% of gate height

            num_passed = torch.sum((robot_position[:, 1] > self.task.gate_position[:, 1]) & 
                                 (torch.abs(robot_position[:, 0] - self.task.gate_position[:, 0]) < curriculum_width_tolerance) &
                                 (robot_position[:, 2] > curriculum_min_height) & (robot_position[:, 2] < curriculum_max_height)).item()

            if num_passed > 0:
                logger.warning(f"  🎉 GATE PASSAGES:          {num_passed}/16 environments!")
                logger.warning(f"  💰 Gate Passage Reward:   {self.task.task_config.reward_parameters['gate_passage_reward_magnitude'].item():.1f} per passage")

            # Check for crashes
            num_crashes = torch.sum(obs_dict["crashes"]).item()
            if num_crashes > 0:
                logger.warning(f"  💥 CRASHES:                {num_crashes}/16 environments")
                logger.warning(f"  💸 Collision Penalty:     {self.task.task_config.reward_parameters['collision_penalty'].item():.1f} per crash")

            # EPISODE-LEVEL REWARD BREAKDOWN: Show how components contribute to episode totals
            if len(self.task.completed_episodes) > 0:
                logger.warning("-"*80)
                logger.warning(f"📈 EPISODE REWARD ANALYSIS (Last {len(self.task.completed_episodes)} Episodes):")

                # Calculate averages across completed episodes
                avg_episode_data = {}
                for key in self.task.completed_episodes[0].keys():
                    avg_episode_data[key] = sum(ep[key] for ep in self.task.completed_episodes) / len(self.task.completed_episodes)

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
                if len(self.task.completed_episodes) >= 5:
                    recent_total = sum(ep['total_reward'] for ep in self.task.completed_episodes[-3:]) / 3
                    older_total = sum(ep['total_reward'] for ep in self.task.completed_episodes[:3]) / 3
                    trend = recent_total - older_total
                    trend_emoji = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                    logger.warning(f"  {trend_emoji} Recent Trend:         {trend:+.1f} (last 3 vs first 3)")

            # CURRENT EPISODE PROGRESS: Show cumulative rewards for ongoing episodes
            logger.warning("-"*80)
            logger.warning("🔄 CURRENT EPISODE PROGRESS (Cumulative):")

            # Average current episode progress across all environments
            avg_current_pos = torch.mean(self.task.episode_pos_reward).item()
            avg_current_very_close = torch.mean(self.task.episode_very_close_reward).item()
            avg_current_getting_closer = torch.mean(self.task.episode_getting_closer_reward).item()
            avg_current_gate_approach = torch.mean(self.task.episode_gate_approach_reward).item()
            avg_current_gate_alignment = torch.mean(self.task.episode_gate_alignment_reward).item()
            avg_current_camera_facing = torch.mean(self.task.episode_camera_facing_reward).item()
            avg_current_action_penalty = torch.mean(self.task.episode_action_penalty).item()
            try:
                avg_current_boundary_penalty = torch.mean(self.task.episode_boundary_violation_penalty).item()
            except RuntimeError:
                avg_current_boundary_penalty = 0.0
            avg_current_collision_penalty = torch.mean(self.task.episode_collision_penalty).item()
            avg_current_episode_length = torch.mean(self.task.episode_lengths).item()

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
            thr = float(self.task.task_config.reward_outlier_threshold)
        except (ValueError, TypeError):
            thr = -180.0
        try:
            if torch.any(rewards < thr):
                _bad = torch.nonzero(rewards < thr, as_tuple=False).squeeze(-1)
                if _bad.numel() > 0:
                    _limit = int(self.task.task_config.reward_outlier_log_limit_per_step)
                    _s = _bad[:_limit]
                    logger.warning(f"[RewardOutlier] envs={_s.tolist()} rewards={rewards[_s].tolist()} crashes={crashes[_s].tolist()}")
                    _dist = torch.norm(self.task.pos_error_vehicle_frame[_s], dim=1)
                    _y = obs_dict['robot_position'][_s, 1]
                    # Also log boundary violation mask to see if it caused spikes
                    try:
                        _bv = boundary_violation_one_shot_mask[_s].tolist()
                    except RuntimeError:
                        _bv = []
                    logger.warning(f"[RewardOutlier] dist={_dist.tolist()} y={_y.tolist()} boundary_violation={_bv}")
        except (ValueError, TypeError):
            pass



