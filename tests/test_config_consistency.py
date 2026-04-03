"""Cross-config consistency tests."""

import isaacgym  # noqa: F401


class TestGateConfigInternalConsistency:
    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config

        self.cfg = task_config

    def test_obs_dim_matches_layout(self):
        assert self.cfg.observation_space_dim == 3 + 3 + 3 + 3 + 3 + 3 + 4 + 64 + 64

    def test_action_dim_is_4(self):
        assert self.cfg.action_space_dim == 4

    def test_curriculum_min_less_than_max(self):
        assert self.cfg.curriculum.min_level < self.cfg.curriculum.max_level

    def test_success_threshold_greater_than_decrease(self):
        assert (
            self.cfg.curriculum.success_rate_for_increase
            > self.cfg.curriculum.success_rate_for_decrease
        )

    def test_vae_latent_dims_fit_obs(self):
        assert self.cfg.vae_config.latent_dims * 2 <= self.cfg.observation_space_dim

    def test_episode_len_positive(self):
        assert self.cfg.episode_len_steps > 0

    def test_collision_penalty_negative(self):
        assert self.cfg.reward_parameters["collision_penalty"] < 0

    def test_gate_passage_positive(self):
        assert self.cfg.reward_parameters["gate_passage_reward_magnitude"] > 0

    def test_reward_params_all_finite(self):
        import math

        for k, v in self.cfg.reward_parameters.items():
            assert math.isfinite(v), f"{k} is not finite: {v}"


class TestRegistryConfigConsistency:
    def test_gate_task_uses_gate_env(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config

        assert task_config.env_name == "gate_env"

    def test_gate_task_uses_lmf2(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config

        assert task_config.robot_name == "lmf2"

    def test_nav_task_uses_obstacles_env(self):
        from aerial_gym.config.task_config.navigation_task_config import task_config

        assert task_config.env_name == "env_with_obstacles"

    def test_both_configs_have_vae(self):
        from aerial_gym.config.task_config.navigation_task_config import task_config as nav
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config as gate

        assert gate.vae_config.use_vae is True
        assert nav.vae_config.use_vae is True
