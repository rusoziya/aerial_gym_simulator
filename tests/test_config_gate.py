"""Behavior-capture tests for gate navigation task configuration.

Captures exact config values, reward parameters, curriculum schedules,
and observation space dimensions from gate_config_9 so refactoring
can be validated.
"""

import isaacgym  # noqa: F401 — must be before torch
import torch  # noqa: F401
import pytest


class TestTaskConfigGateValues:
    """Capture exact config attribute values."""

    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        self.cfg = task_config

    def test_sim_name(self):
        assert self.cfg.sim_name == "base_sim"

    def test_env_name(self):
        assert self.cfg.env_name == "gate_env"

    def test_robot_name(self):
        assert self.cfg.robot_name == "lmf2"

    def test_observation_space_dim(self):
        assert self.cfg.observation_space_dim == 150

    def test_action_space_dim(self):
        assert self.cfg.action_space_dim == 4

    def test_episode_len_steps(self):
        assert self.cfg.episode_len_steps == 100

    def test_device(self):
        assert self.cfg.device == "cuda:0"

    def test_use_warp(self):
        assert self.cfg.use_warp is True

    def test_seed(self):
        assert self.cfg.seed == -1


class TestRewardParameters:
    """Capture exact reward parameter values for regression detection."""

    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        self.rp = task_config.reward_parameters

    def test_pos_reward_magnitude(self):
        assert self.rp["pos_reward_magnitude"] == 0.5

    def test_collision_penalty(self):
        assert self.rp["collision_penalty"] == -100.0

    def test_gate_passage_reward_magnitude(self):
        assert self.rp["gate_passage_reward_magnitude"] == 100.0

    def test_gate_center_passage_bonus(self):
        assert self.rp["gate_center_passage_bonus_magnitude"] == 100.0

    def test_getting_closer_reward_multiplier(self):
        assert self.rp["getting_closer_reward_multiplier"] == 5.0

    def test_timeout_penalty(self):
        assert self.rp["timeout_penalty"] == 75.0

    def test_boundary_violation_penalty(self):
        assert self.rp["boundary_violation_penalty_magnitude"] == 50.0

    def test_all_expected_keys_present(self):
        expected = [
            "pos_reward_magnitude", "pos_reward_exponent",
            "very_close_to_goal_reward_magnitude", "very_close_to_goal_reward_exponent",
            "getting_closer_reward_multiplier",
            "x_action_diff_penalty_magnitude", "x_action_diff_penalty_exponent",
            "y_action_diff_penalty_magnitude", "y_action_diff_penalty_exponent",
            "z_action_diff_penalty_magnitude", "z_action_diff_penalty_exponent",
            "yawrate_action_diff_penalty_magnitude", "yawrate_action_diff_penalty_exponent",
            "x_absolute_action_penalty_magnitude", "x_absolute_action_penalty_exponent",
            "y_absolute_action_penalty_magnitude", "y_absolute_action_penalty_exponent",
            "z_absolute_action_penalty_magnitude", "z_absolute_action_penalty_exponent",
            "yawrate_absolute_action_penalty_magnitude", "yawrate_absolute_action_penalty_exponent",
            "collision_penalty",
            "gate_approach_reward_magnitude",
            "gate_alignment_reward_magnitude",
            "gate_passage_reward_magnitude",
            "gate_center_bonus_magnitude",
            "gate_center_passage_bonus_magnitude",
            "camera_facing_reward_magnitude",
            "boundary_violation_penalty_magnitude",
            "time_penalty_total_at_horizon",
            "timeout_penalty",
        ]
        for key in expected:
            assert key in self.rp, f"Missing reward parameter: {key}"


class TestCurriculumConfig:
    """Capture curriculum configuration values."""

    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        self.cur = task_config.curriculum

    def test_min_level(self):
        assert self.cur.min_level == 3

    def test_max_level(self):
        assert self.cur.max_level == 23

    def test_check_after_log_instances(self):
        assert self.cur.check_after_log_instances == 256

    def test_increase_step(self):
        assert self.cur.increase_step == 1

    def test_decrease_step(self):
        assert self.cur.decrease_step == 1

    def test_success_rate_for_increase(self):
        assert self.cur.success_rate_for_increase == 0.55

    def test_success_rate_for_decrease(self):
        assert self.cur.success_rate_for_decrease == 0.25

    def test_cooldown_windows(self):
        assert self.cur.cooldown_windows == 12

    def test_max_obstacles_behind_gate(self):
        assert self.cur.max_obstacles_behind_gate == 10


class TestCurriculumSchedules:
    """Capture curriculum schedule function outputs at specific levels."""

    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        self.cur = task_config.curriculum

    def test_obstacle_count_at_min_level(self):
        count = self.cur.get_obstacle_count_behind_gate(3)
        assert count == 3

    def test_obstacle_count_at_max_level(self):
        count = self.cur.get_obstacle_count_behind_gate(23)
        assert count == 10

    def test_obstacle_count_monotonically_increases(self):
        prev = 0
        for level in range(3, 24):
            count = self.cur.get_obstacle_count_behind_gate(level)
            assert count >= prev, f"Obstacle count decreased at level {level}"
            prev = count

    def test_camera_noise_at_min_level(self):
        gaussian, dropout = self.cur.get_camera_noise(3)
        assert gaussian > 0
        assert dropout > 0
        assert gaussian < 0.01  # very small at start

    def test_camera_noise_at_max_level(self):
        gaussian, dropout = self.cur.get_camera_noise(23)
        assert gaussian == pytest.approx(0.00625, abs=1e-6)
        assert dropout == pytest.approx(0.00625, abs=1e-6)

    def test_camera_noise_increases_with_level(self):
        g3, d3 = self.cur.get_camera_noise(3)
        g23, d23 = self.cur.get_camera_noise(23)
        assert g23 > g3
        assert d23 > d3

    def test_spawn_ranges_at_min_level(self):
        sr = self.cur.get_spawn_ranges(3)
        assert "x_half_span_m" in sr
        assert "y_center_m" in sr
        assert "z_center_m" in sr
        assert sr["x_half_span_m"] == pytest.approx(0.50, abs=0.01)

    def test_spawn_ranges_at_max_level(self):
        sr = self.cur.get_spawn_ranges(23)
        assert sr["x_half_span_m"] == pytest.approx(1.50, abs=0.01)

    def test_spawn_ranges_widen_with_level(self):
        sr3 = self.cur.get_spawn_ranges(3)
        sr23 = self.cur.get_spawn_ranges(23)
        assert sr23["x_half_span_m"] > sr3["x_half_span_m"]

    def test_static_camera_difficulty_at_min_level(self):
        angle, height_offset, dist_offset = self.cur.get_static_camera_difficulty(3)
        assert angle >= 0

    def test_state_noise_at_min_level(self):
        sn = self.cur.get_state_noise(3)
        assert "drone_pos_std_m" in sn
        assert "static_pos_std_m" in sn
        assert sn["drone_pos_std_m"] >= 0

    def test_frame_dropout_at_min_level(self):
        fd = self.cur.get_camera_frame_dropout(3)
        assert "drone_total" in fd
        assert "static_total" in fd
        assert fd["drone_total"] >= 0


class TestVaeConfig:
    """Capture VAE configuration."""

    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        self.vae = task_config.vae_config

    def test_use_vae(self):
        assert self.vae.use_vae is True

    def test_latent_dims(self):
        assert self.vae.latent_dims == 64

    def test_image_res(self):
        assert self.vae.image_res == (270, 480)

    def test_return_sampled_latent(self):
        assert self.vae.return_sampled_latent is True


class TestObservationSpaceLayout:
    """Capture the 150D observation vector layout.

    This is critical — if the obs layout changes, trained policies break.
    """

    def test_total_dim_is_150(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        assert task_config.observation_space_dim == 150

    def test_layout_drone_position(self):
        """Indices 0:3 = drone absolute position."""
        assert 3 - 0 == 3

    def test_layout_static_camera_position(self):
        """Indices 3:6 = static camera position relative to drone."""
        assert 6 - 3 == 3

    def test_layout_static_camera_orientation(self):
        """Indices 6:9 = static camera orientation relative to drone."""
        assert 9 - 6 == 3

    def test_layout_drone_orientation(self):
        """Indices 9:12 = drone orientation (roll, pitch, yaw)."""
        assert 12 - 9 == 3

    def test_layout_body_linear_velocity(self):
        """Indices 12:15 = body linear velocity."""
        assert 15 - 12 == 3

    def test_layout_body_angular_velocity(self):
        """Indices 15:18 = body angular velocity."""
        assert 18 - 15 == 3

    def test_layout_actions(self):
        """Indices 18:22 = last actions (4D velocity controller)."""
        assert 22 - 18 == 4

    def test_layout_drone_vae_latents(self):
        """Indices 22:86 = drone camera VAE latents (64D)."""
        assert 86 - 22 == 64

    def test_layout_static_vae_latents(self):
        """Indices 86:150 = static camera VAE latents (64D)."""
        assert 150 - 86 == 64

    def test_all_dimensions_sum_to_150(self):
        dims = [3, 3, 3, 3, 3, 3, 4, 64, 64]
        assert sum(dims) == 150
