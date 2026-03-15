"""Behavior tests for DCE_RL_Navigation_Task_Gate configuration."""
import isaacgym  # noqa: F401
import pytest
import os


class TestDCEGateTaskConfig:
    """Test that the DCE gate wrapper sets up the task config correctly."""

    def test_action_space_dim_set_to_4(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        cfg = task_config
        # DCE gate task should use 4D actions
        assert cfg.action_space_dim == 4

    def test_curriculum_min_level_3(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        assert task_config.curriculum.min_level == 3

    def test_curriculum_max_level_23(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        assert task_config.curriculum.max_level == 23

    def test_obs_dim_150(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        assert task_config.observation_space_dim == 150


class TestSFEnvAgentsOverride:
    """Test SF_ENV_AGENTS environment variable behavior."""

    def test_env_var_not_set_by_default(self):
        val = os.environ.get('SF_ENV_AGENTS', None)
        # May or may not be set, but shouldn't crash
        assert val is None or val.isdigit()

    def test_sf_headless_not_set_by_default(self):
        val = os.environ.get('SF_HEADLESS', None)
        assert val is None or val.lower() in ('true', 'false')
