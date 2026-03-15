"""Behavior-capture tests for action transformation (gate_config_10).

The gate task uses a 4D velocity controller action transformation
that maps [-1,1] actions to velocity commands.
"""
import isaacgym  # noqa: F401
import torch
import pytest


class TestActionTransformation:
    """Test the action_transformation_function from gate config."""

    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        self.transform = task_config.action_transformation_function

    def test_returns_same_shape(self):
        actions = torch.randn(4, 4)
        result = self.transform(actions)
        assert result.shape == actions.shape

    def test_zero_input_zero_output(self):
        actions = torch.zeros(4, 4)
        result = self.transform(actions)
        assert torch.allclose(result, torch.zeros(4, 4), atol=1e-5)

    def test_batch_dimension_preserved(self):
        for n in [1, 4, 16, 64]:
            actions = torch.randn(n, 4)
            result = self.transform(actions)
            assert result.shape[0] == n
            assert result.shape[1] == 4

    def test_output_is_finite(self):
        actions = torch.randn(8, 4)
        result = self.transform(actions)
        assert torch.all(torch.isfinite(result))

    @pytest.mark.parametrize("dim", [0, 1, 2, 3])
    def test_each_dimension_responds(self, dim):
        actions = torch.zeros(1, 4)
        actions[0, dim] = 1.0
        result = self.transform(actions)
        assert result[0, dim].abs().item() > 0


class TestBaseNavigationConfig:
    """Test base navigation task config values."""

    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config import task_config
        self.cfg = task_config

    def test_sim_name(self):
        assert self.cfg.sim_name == "base_sim"

    def test_env_name(self):
        assert self.cfg.env_name == "env_with_obstacles"

    def test_observation_space_dim(self):
        assert self.cfg.observation_space_dim == 81

    def test_action_space_dim(self):
        assert self.cfg.action_space_dim == 4

    def test_has_reward_parameters(self):
        assert isinstance(self.cfg.reward_parameters, dict)
        assert len(self.cfg.reward_parameters) > 0

    def test_has_curriculum(self):
        assert self.cfg.curriculum is not None
        assert self.cfg.curriculum.min_level is not None
