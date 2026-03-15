"""Behavior tests for observation transformation in SF pipeline."""
import isaacgym  # noqa: F401
import torch
import pytest
from unittest.mock import Mock


class TestObsTransformBaseWrapper:
    """Test how the base wrapper transforms observations."""

    def test_observations_key_remapped_to_obs(self):
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import AerialGymVecEnv
        mock_env = Mock()
        mock_env.num_envs = 2
        mock_env.reset.return_value = (
            {"observations": torch.tensor([[1.0]*81, [2.0]*81])},
            torch.zeros(2), torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool), {}
        )
        wrapper = AerialGymVecEnv(mock_env, "obs")
        obs, _ = wrapper.reset()
        assert "obs" in obs
        assert "observations" not in obs
        assert obs["obs"][0, 0].item() == pytest.approx(1.0)
        assert obs["obs"][1, 0].item() == pytest.approx(2.0)

    def test_step_preserves_reward_shape(self):
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import AerialGymVecEnv
        mock_env = Mock()
        mock_env.num_envs = 4
        mock_env.step.return_value = (
            {"observations": torch.zeros(4, 81)},
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.zeros(4, dtype=torch.bool),
            torch.zeros(4, dtype=torch.bool), {}
        )
        wrapper = AerialGymVecEnv(mock_env, "obs")
        _, rew, _, _, _ = wrapper.step(torch.zeros(4, 3))
        assert rew.shape == (4,)
        assert rew[0].item() == pytest.approx(1.0)
        assert rew[3].item() == pytest.approx(4.0)

    def test_step_preserves_termination_flags(self):
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import AerialGymVecEnv
        mock_env = Mock()
        mock_env.num_envs = 4
        mock_env.step.return_value = (
            {"observations": torch.zeros(4, 81)},
            torch.zeros(4),
            torch.tensor([True, False, True, False]),
            torch.tensor([False, True, False, True]), {}
        )
        wrapper = AerialGymVecEnv(mock_env, "obs")
        _, _, term, trunc, _ = wrapper.step(torch.zeros(4, 3))
        assert term[0].item() is True
        assert term[1].item() is False
        assert trunc[1].item() is True

    def test_obs_shape_preserved_through_pipeline(self):
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import AerialGymVecEnv
        mock_env = Mock()
        mock_env.num_envs = 8
        obs_data = torch.randn(8, 81)
        mock_env.step.return_value = (
            {"observations": obs_data},
            torch.zeros(8), torch.zeros(8, dtype=torch.bool),
            torch.zeros(8, dtype=torch.bool), {}
        )
        wrapper = AerialGymVecEnv(mock_env, "obs")
        obs, _, _, _, _ = wrapper.step(torch.zeros(8, 3))
        assert obs["obs"].shape == (8, 81)
        assert torch.allclose(obs["obs"], obs_data)
