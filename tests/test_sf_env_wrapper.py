"""Behavior tests for Sample Factory AerialGymVecEnv wrapper."""
import isaacgym  # noqa: F401
import torch
import pytest
from unittest.mock import Mock, MagicMock
import gymnasium as gym
import numpy as np


class TestAerialGymVecEnvInit:
    """Test the SF env wrapper initialization behavior."""

    def _make_mock_env(self, num_envs=4, obs_dim=81):
        mock = Mock()
        mock.num_envs = num_envs
        mock.reset.return_value = (
            {"observations": torch.zeros(num_envs, obs_dim), "image_obs": torch.zeros(num_envs, 1, 270, 480)},
            torch.zeros(num_envs), torch.zeros(num_envs, dtype=torch.bool),
            torch.zeros(num_envs, dtype=torch.bool), {}
        )
        mock.step.return_value = (
            {"observations": torch.randn(num_envs, obs_dim), "image_obs": torch.zeros(num_envs, 1, 270, 480)},
            torch.randn(num_envs), torch.zeros(num_envs, dtype=torch.bool),
            torch.zeros(num_envs, dtype=torch.bool), {}
        )
        return mock

    def test_num_agents_matches_envs(self):
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import AerialGymVecEnv
        mock_env = self._make_mock_env(num_envs=8)
        wrapper = AerialGymVecEnv(mock_env, "obs")
        assert wrapper.num_agents == 8

    def test_is_multiagent_true(self):
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import AerialGymVecEnv
        mock_env = self._make_mock_env()
        wrapper = AerialGymVecEnv(mock_env, "obs")
        assert wrapper.is_multiagent is True

    def test_action_space_is_3d(self):
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import AerialGymVecEnv
        mock_env = self._make_mock_env()
        wrapper = AerialGymVecEnv(mock_env, "obs")
        assert wrapper.action_space.shape == (3,)

    def test_obs_space_has_obs_key(self):
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import AerialGymVecEnv
        mock_env = self._make_mock_env()
        wrapper = AerialGymVecEnv(mock_env, "obs")
        assert "obs" in wrapper.observation_space.spaces

    def test_obs_space_shape_81(self):
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import AerialGymVecEnv
        mock_env = self._make_mock_env(obs_dim=81)
        wrapper = AerialGymVecEnv(mock_env, "obs")
        assert wrapper.observation_space["obs"].shape == (81,)

    def test_reset_returns_obs_dict_and_infos(self):
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import AerialGymVecEnv
        mock_env = self._make_mock_env()
        wrapper = AerialGymVecEnv(mock_env, "obs")
        obs, infos = wrapper.reset()
        assert "obs" in obs
        assert obs["obs"].shape[0] == 4

    def test_step_returns_5_tuple(self):
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import AerialGymVecEnv
        mock_env = self._make_mock_env()
        wrapper = AerialGymVecEnv(mock_env, "obs")
        action = torch.zeros(4, 3)
        result = wrapper.step(action)
        assert len(result) == 5
        obs, rew, term, trunc, infos = result
        assert "obs" in obs

    def test_step_passes_action_to_env(self):
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import AerialGymVecEnv
        mock_env = self._make_mock_env()
        wrapper = AerialGymVecEnv(mock_env, "obs")
        action = torch.ones(4, 3) * 0.5
        wrapper.step(action)
        mock_env.step.assert_called_once()


class TestGateEnvWrapperInit:
    """Test the gate-specific SF wrapper."""

    def _make_mock_gate_env(self, num_envs=4):
        mock = Mock()
        mock.num_envs = num_envs
        mock.task_config = Mock()
        mock.task_config.observation_space_dim = 150
        mock.reset.return_value = (
            {"observations": torch.zeros(num_envs, 150), "image_obs": torch.zeros(num_envs, 1, 270, 480)},
            torch.zeros(num_envs), torch.zeros(num_envs, dtype=torch.bool),
            torch.zeros(num_envs, dtype=torch.bool), {}
        )
        mock.step.return_value = (
            {"observations": torch.randn(num_envs, 150), "image_obs": torch.zeros(num_envs, 1, 270, 480)},
            torch.randn(num_envs), torch.zeros(num_envs, dtype=torch.bool),
            torch.zeros(num_envs, dtype=torch.bool), {}
        )
        return mock

    def test_gate_action_space_4d(self):
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net_gate import AerialGymVecEnv
        mock_env = self._make_mock_gate_env()
        wrapper = AerialGymVecEnv(mock_env, "obs")
        assert wrapper.action_space.shape == (4,)

    def test_gate_obs_150d(self):
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net_gate import AerialGymVecEnv
        mock_env = self._make_mock_gate_env()
        wrapper = AerialGymVecEnv(mock_env, "obs")
        assert wrapper.observation_space["obs"].shape == (150,)
