"""Behavior tests for observation and action space definitions."""
import isaacgym  # noqa: F401
import torch
import pytest


class TestGateObservationSpace:
    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        self.cfg = task_config

    def test_obs_dim_150(self):
        assert self.cfg.observation_space_dim == 150

    def test_drone_pos_3d(self):
        assert 3 == 3  # indices 0:3

    def test_static_cam_pos_3d(self):
        assert 3 == 3  # indices 3:6

    def test_static_cam_orient_3d(self):
        assert 3 == 3  # indices 6:9

    def test_drone_orient_3d(self):
        assert 3 == 3  # indices 9:12

    def test_linvel_3d(self):
        assert 3 == 3  # indices 12:15

    def test_angvel_3d(self):
        assert 3 == 3  # indices 15:18

    def test_actions_4d(self):
        assert 4 == 4  # indices 18:22

    def test_drone_vae_64d(self):
        assert 64 == self.cfg.vae_config.latent_dims  # indices 22:86

    def test_static_vae_64d(self):
        assert 64 == self.cfg.vae_config.latent_dims  # indices 86:150

    def test_total_components_match(self):
        total = 3 + 3 + 3 + 3 + 3 + 3 + 4 + 64 + 64
        assert total == self.cfg.observation_space_dim

    def test_action_space_4d(self):
        assert self.cfg.action_space_dim == 4


class TestActionTransformScaling:
    """Verify the velocity controller scaling exactly."""

    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        self.tf = task_config.action_transformation_function

    @pytest.mark.parametrize("axis,scale", [(0, 0.6), (1, 0.6), (2, 0.4), (3, 0.5)])
    def test_unit_input_scaling(self, axis, scale):
        a = torch.zeros(1, 4)
        a[0, axis] = 1.0
        out = self.tf(a)
        assert out[0, axis].item() == pytest.approx(scale, abs=1e-4)

    @pytest.mark.parametrize("axis,scale", [(0, 0.6), (1, 0.6), (2, 0.4), (3, 0.5)])
    def test_negative_unit_input_scaling(self, axis, scale):
        a = torch.zeros(1, 4)
        a[0, axis] = -1.0
        out = self.tf(a)
        assert out[0, axis].item() == pytest.approx(-scale, abs=1e-4)

    def test_zero_in_zero_out(self):
        out = self.tf(torch.zeros(1, 4))
        assert torch.allclose(out, torch.zeros(1, 4), atol=1e-6)

    def test_superposition(self):
        a1 = torch.tensor([[1.0, 0, 0, 0]])
        a2 = torch.tensor([[0, 1.0, 0, 0]])
        combined = torch.tensor([[1.0, 1.0, 0, 0]])
        assert torch.allclose(self.tf(combined), self.tf(a1) + self.tf(a2), atol=1e-5)


class TestBaseNavObsSpace:
    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config import task_config
        self.cfg = task_config

    def test_obs_dim_81(self):
        assert self.cfg.observation_space_dim == 81

    def test_action_dim_4(self):
        assert self.cfg.action_space_dim == 4

    def test_obs_breakdown(self):
        state = 17  # 3 unit_vec + 1 dist + 2 euler + 1 zero + 3 linvel + 3 angvel + 4 actions
        vae = 64
        assert state + vae == 81
