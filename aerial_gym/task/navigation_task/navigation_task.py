from __future__ import annotations

from typing import Any

from aerial_gym.task.base_navigation_task import BaseNavigationTask
import torch
import os

from aerial_gym.utils.math import *  # noqa: F401,F403
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("navigation_task")


class NavigationTask(BaseNavigationTask):
    def __init__(
        self,
        task_config: Any,
        seed: int | None = None,
        num_envs: int | None = None,
        headless: bool | None = None,
        device: str | None = None,
        use_warp: bool | None = None,
    ) -> None:
        super().__init__(task_config, seed, num_envs, headless, device, use_warp)

        # Position-setpoint navigation uses random target ratios within env bounds
        self.target_min_ratio = torch.tensor(
            self.task_config.target_min_ratio, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.target_max_ratio = torch.tensor(
            self.task_config.target_max_ratio, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)

        self.obs_dict["num_obstacles_in_env"] = self.curriculum_level

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        target_ratio = torch_rand_float_tensor(self.target_min_ratio, self.target_max_ratio)
        self.target_position[env_ids] = torch_interpolate_ratio(
            min=self.obs_dict["env_bounds_min"][env_ids],
            max=self.obs_dict["env_bounds_max"][env_ids],
            ratio=target_ratio[env_ids],
        )
        self.infos = {}

    def step(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        transformed_action = self.action_transformation_function(actions)
        self.sim_env.step(actions=transformed_action)

        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        successes = self.truncations * (
            torch.norm(self.target_position - self.obs_dict["robot_position"], dim=1) < 1.0
        )
        successes = torch.where(self.terminations > 0, torch.zeros_like(successes), successes)
        timeouts = torch.where(
            self.truncations > 0, torch.logical_not(successes), torch.zeros_like(successes)
        )
        timeouts = torch.where(self.terminations > 0, torch.zeros_like(timeouts), timeouts)

        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = self.terminations

        self.infos.setdefault("episode_extra_stats", {})
        self.infos["episode_extra_stats"]["curriculum/current_level"] = float(
            self.curriculum_level
        )
        self.infos["episode_extra_stats"]["curriculum/current_progress"] = float(
            self.curriculum_progress_fraction
        )

        self.logging_sanity_check(self.infos)
        self.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )

        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)
        self.num_task_steps += 1

        self.process_image_observation()
        self.post_image_reward_addition()

        if not self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()
        return return_tuple

    def post_image_reward_addition(self) -> None:
        image_obs = 10.0 * self.obs_dict["depth_range_pixels"].squeeze(1)
        image_obs[image_obs < 0] = 10.0
        self.min_pixel_dist = torch.amin(image_obs, dim=(1, 2))
        self.rewards[~self.terminations] += -exponential_reward_function(
            4.0, 1.0, self.min_pixel_dist[~self.terminations]
        )

    def process_obs_for_task(self) -> None:
        vec_to_tgt = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )
        perturbed_vec_to_tgt = vec_to_tgt + 0.1 * 2 * (torch.rand_like(vec_to_tgt - 0.5))
        dist_to_tgt = torch.norm(vec_to_tgt, dim=-1)
        perturbed_unit_vec_to_tgt = perturbed_vec_to_tgt / dist_to_tgt.unsqueeze(1)
        self.task_obs["observations"][:, 0:3] = perturbed_unit_vec_to_tgt
        self.task_obs["observations"][:, 3] = dist_to_tgt
        euler_angles = ssa(self.obs_dict["robot_euler_angles"])
        perturbed_euler_angles = euler_angles + 0.1 * (torch.rand_like(euler_angles) - 0.5)
        self.task_obs["observations"][:, 4] = perturbed_euler_angles[:, 0]
        self.task_obs["observations"][:, 5] = perturbed_euler_angles[:, 1]
        self.task_obs["observations"][:, 6] = 0.0
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 13:17] = self.obs_dict["robot_actions"]
        if self.task_config.vae_config.use_vae:
            self.task_obs["observations"][:, 17:] = self.image_latents
        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations
        self.task_obs["image_obs"] = self.obs_dict["depth_range_pixels"]

    def compute_rewards_and_crashes(self, obs_dict: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        self.pos_error_vehicle_frame_prev[:] = self.pos_error_vehicle_frame
        self.pos_error_vehicle_frame[:] = quat_rotate_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        disable_cm = (
            str(os.environ.get("SF_DISABLE_CURRICULUM_MULTIPLIER", "false")).lower() == "true"
        )
        frac_eff = 0.0 if disable_cm else float(self.curriculum_progress_fraction)

        return compute_reward(
            self.pos_error_vehicle_frame,
            self.pos_error_vehicle_frame_prev,
            obs_dict["crashes"],
            obs_dict["robot_actions"],
            obs_dict["robot_prev_actions"],
            frac_eff,
            self.task_config.reward_parameters,
        )

    def check_and_update_curriculum_level(self, successes: torch.Tensor, crashes: torch.Tensor, timeouts: torch.Tensor) -> None:
        self.success_aggregate += torch.sum(successes)
        self.crashes_aggregate += torch.sum(crashes)
        self.timeouts_aggregate += torch.sum(timeouts)

        instances = self.success_aggregate + self.crashes_aggregate + self.timeouts_aggregate

        if instances >= self.task_config.curriculum.check_after_log_instances:
            success_rate = self.success_aggregate / instances
            crash_rate = self.crashes_aggregate / instances
            timeout_rate = self.timeouts_aggregate / instances

            if success_rate > self.task_config.curriculum.success_rate_for_increase:
                self.curriculum_level += self.task_config.curriculum.increase_step
            elif success_rate < self.task_config.curriculum.success_rate_for_decrease:
                self.curriculum_level -= self.task_config.curriculum.decrease_step

            self.curriculum_level = min(
                max(self.curriculum_level, self.task_config.curriculum.min_level),
                self.task_config.curriculum.max_level,
            )
            self.obs_dict["curriculum_level"] = self.curriculum_level
            self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
            self.curriculum_progress_fraction = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

            logger.warning(
                f"Curriculum Level: {self.curriculum_level}, "
                f"Progress: {self.curriculum_progress_fraction}"
            )
            logger.warning(
                f"SR: {success_rate}, CR: {crash_rate}, TR: {timeout_rate}"
            )

            self.infos["curriculum/level"] = torch.tensor(
                self.curriculum_level, dtype=torch.float32
            )
            self.infos["curriculum/progress"] = torch.tensor(
                self.curriculum_progress_fraction, dtype=torch.float32
            )
            self.infos["curriculum/success_rate"] = torch.tensor(
                success_rate, dtype=torch.float32
            )
            self.infos["curriculum/crash_rate"] = torch.tensor(
                crash_rate, dtype=torch.float32
            )
            self.infos["curriculum/timeout_rate"] = torch.tensor(
                timeout_rate, dtype=torch.float32
            )

            self.success_aggregate = 0
            self.crashes_aggregate = 0
            self.timeouts_aggregate = 0


@torch.jit.script
def compute_reward(
    pos_error,
    prev_pos_error,
    crashes,
    action,
    prev_action,
    curriculum_progress_fraction,
    parameter_dict,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]
    MULTIPLICATION_FACTOR_REWARD = 1.0 + (0.5) * curriculum_progress_fraction
    dist = torch.norm(pos_error, dim=1)
    prev_dist_to_goal = torch.norm(prev_pos_error, dim=1)
    pos_reward = exponential_reward_function(
        parameter_dict["pos_reward_magnitude"],
        parameter_dict["pos_reward_exponent"],
        dist,
    )
    very_close_to_goal_reward = exponential_reward_function(
        parameter_dict["very_close_to_goal_reward_magnitude"],
        parameter_dict["very_close_to_goal_reward_exponent"],
        dist,
    )

    getting_closer = prev_dist_to_goal - dist
    getting_closer_reward = torch.where(
        getting_closer > 0,
        parameter_dict["getting_closer_reward_multiplier"] * getting_closer,
        2.0 * parameter_dict["getting_closer_reward_multiplier"] * getting_closer,
    )

    distance_from_goal_reward = (20.0 - dist) / 20.0
    action_diff = action - prev_action
    x_diff_penalty = exponential_penalty_function(
        parameter_dict["x_action_diff_penalty_magnitude"],
        parameter_dict["x_action_diff_penalty_exponent"],
        action_diff[:, 0],
    )
    z_diff_penalty = exponential_penalty_function(
        parameter_dict["z_action_diff_penalty_magnitude"],
        parameter_dict["z_action_diff_penalty_exponent"],
        action_diff[:, 2],
    )
    yawrate_diff_penalty = exponential_penalty_function(
        parameter_dict["yawrate_action_diff_penalty_magnitude"],
        parameter_dict["yawrate_action_diff_penalty_exponent"],
        action_diff[:, 3],
    )
    action_diff_penalty = x_diff_penalty + z_diff_penalty + yawrate_diff_penalty
    x_absolute_penalty = exponential_penalty_function(
        parameter_dict["x_absolute_action_penalty_magnitude"],
        parameter_dict["x_absolute_action_penalty_exponent"],
        action[:, 0],
    )
    z_absolute_penalty = exponential_penalty_function(
        parameter_dict["z_absolute_action_penalty_magnitude"],
        parameter_dict["z_absolute_action_penalty_exponent"],
        action[:, 2],
    )
    yawrate_absolute_penalty = exponential_penalty_function(
        parameter_dict["yawrate_absolute_action_penalty_magnitude"],
        parameter_dict["yawrate_absolute_action_penalty_exponent"],
        action[:, 3],
    )
    absolute_action_penalty = x_absolute_penalty + z_absolute_penalty + yawrate_absolute_penalty
    total_action_penalty = action_diff_penalty + absolute_action_penalty

    reward = (
        MULTIPLICATION_FACTOR_REWARD
        * (
            pos_reward
            + very_close_to_goal_reward
            + getting_closer_reward
            + distance_from_goal_reward
        )
        + total_action_penalty
    )

    reward[:] = torch.where(
        crashes > 0,
        parameter_dict["collision_penalty"] * torch.ones_like(reward),
        reward,
    )
    return reward, crashes
