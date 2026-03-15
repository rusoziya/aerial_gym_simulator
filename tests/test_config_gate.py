"""Behavior-capture tests for gate_config_10 configuration values.
These lock down exact values so refactoring drift is detected.
"""
import isaacgym  # noqa: F401
import pytest

class TestTaskConfigValues:
    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        self.cfg = task_config

    def test_sim_name(self): assert self.cfg.sim_name == "base_sim"
    def test_env_name(self): assert self.cfg.env_name == "gate_env"
    def test_robot_name(self): assert self.cfg.robot_name == "lmf2"
    def test_obs_dim(self): assert self.cfg.observation_space_dim == 150
    def test_action_dim(self): assert self.cfg.action_space_dim == 4
    def test_episode_len(self): assert self.cfg.episode_len_steps == 100
    def test_use_warp(self): assert self.cfg.use_warp is True
    def test_seed(self): assert self.cfg.seed == -1

class TestRewardParameters:
    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        self.rp = task_config.reward_parameters

    def test_pos_reward_magnitude(self): assert self.rp["pos_reward_magnitude"] == 0.5
    def test_pos_reward_exponent(self): assert self.rp["pos_reward_exponent"] == 0.15
    def test_very_close_magnitude(self): assert self.rp["very_close_to_goal_reward_magnitude"] == 0.75
    def test_getting_closer(self): assert self.rp["getting_closer_reward_multiplier"] == 5.0
    def test_collision_penalty(self): assert self.rp["collision_penalty"] == -100.0
    def test_gate_passage(self): assert self.rp["gate_passage_reward_magnitude"] == 100.0
    def test_gate_center_passage(self): assert self.rp["gate_center_passage_bonus_magnitude"] == 100.0
    def test_gate_approach(self): assert self.rp["gate_approach_reward_magnitude"] == 1.25
    def test_gate_alignment(self): assert self.rp["gate_alignment_reward_magnitude"] == 0.5
    def test_gate_center_bonus(self): assert self.rp["gate_center_bonus_magnitude"] == 1.25
    def test_timeout_penalty(self): assert self.rp["timeout_penalty"] == 75.0
    def test_boundary_violation(self): assert self.rp["boundary_violation_penalty_magnitude"] == 50.0
    def test_time_penalty_total(self): assert self.rp["time_penalty_total_at_horizon"] == 4.0
    def test_camera_facing_disabled(self): assert self.rp["camera_facing_reward_magnitude"] == 0.0
    def test_image_penalty_magnitude(self): assert self.rp["image_penalty_magnitude"] == 0.2

class TestCurriculumConfig:
    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        self.cur = task_config.curriculum

    def test_min_level(self): assert self.cur.min_level == 3
    def test_max_level(self): assert self.cur.max_level == 23
    def test_check_after(self): assert self.cur.check_after_log_instances == 256
    def test_increase_step(self): assert self.cur.increase_step == 1
    def test_decrease_step(self): assert self.cur.decrease_step == 1
    def test_success_threshold(self): assert self.cur.success_rate_for_increase == 0.55
    def test_decrease_threshold(self): assert self.cur.success_rate_for_decrease == 0.25
    def test_cooldown(self): assert self.cur.cooldown_windows == 12
    def test_max_obstacles(self): assert self.cur.max_obstacles_behind_gate == 10

class TestCurriculumSchedules:
    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        self.cur = task_config.curriculum

    def test_obstacles_at_min(self): assert self.cur.get_obstacle_count_behind_gate(3) == 3
    def test_obstacles_at_max(self): assert self.cur.get_obstacle_count_behind_gate(23) == 10
    def test_obstacles_monotonic(self):
        prev = 0
        for lvl in range(3, 24):
            c = self.cur.get_obstacle_count_behind_gate(lvl)
            assert c >= prev
            prev = c

    def test_camera_noise_at_min(self):
        g, d = self.cur.get_camera_noise(3)
        assert g > 0 and g < 0.001

    def test_camera_noise_at_max(self):
        g, d = self.cur.get_camera_noise(23)
        assert g == pytest.approx(0.00625, abs=1e-6)

    def test_camera_noise_increases(self):
        g3, _ = self.cur.get_camera_noise(3)
        g23, _ = self.cur.get_camera_noise(23)
        assert g23 > g3

    def test_spawn_x_at_min(self):
        sr = self.cur.get_spawn_ranges(3)
        assert sr["x_half_span_m"] == pytest.approx(0.5, abs=0.01)

    def test_spawn_x_at_max(self):
        sr = self.cur.get_spawn_ranges(23)
        assert sr["x_half_span_m"] == pytest.approx(1.5, abs=0.01)

    def test_spawn_widens(self):
        assert self.cur.get_spawn_ranges(23)["x_half_span_m"] > self.cur.get_spawn_ranges(3)["x_half_span_m"]

    def test_state_noise_keys(self):
        sn = self.cur.get_state_noise(3)
        assert "drone_pos_std_m" in sn
        assert "static_pos_std_m" in sn

    def test_frame_dropout_keys(self):
        fd = self.cur.get_camera_frame_dropout(3)
        assert "drone_total" in fd
        assert "static_total" in fd

class TestVaeConfig:
    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        self.vae = task_config.vae_config

    def test_use_vae(self): assert self.vae.use_vae is True
    def test_latent_dims(self): assert self.vae.latent_dims == 64
    def test_image_res(self): assert self.vae.image_res == (270, 480)

class TestObservationLayout:
    def test_total_150(self):
        dims = [3, 3, 3, 3, 3, 3, 4, 64, 64]
        assert sum(dims) == 150
