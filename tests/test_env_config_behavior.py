"""Behavior tests for environment configuration values.

Locks down exact env config values that affect simulation behavior.
"""
import isaacgym  # noqa: F401
import pytest


class TestGateEnvPhysics:
    def setup_method(self):
        from aerial_gym.config.env_config.gate_env import GateEnvCfg
        self.env = GateEnvCfg.env

    def test_env_spacing(self):
        assert self.env.env_spacing == 10.0

    def test_collision_threshold(self):
        assert self.env.collision_force_threshold == pytest.approx(0.05)

    def test_physics_steps_per_env_step(self):
        assert self.env.num_physics_steps_per_env_step_mean == 10

    def test_physics_steps_std_zero(self):
        assert self.env.num_physics_steps_per_env_step_std == 0

    def test_reset_on_collision(self):
        assert self.env.reset_on_collision is True

    def test_no_ground_plane(self):
        assert self.env.create_ground_plane is False

    def test_use_warp(self):
        assert self.env.use_warp is True

    def test_sample_timestep_for_latency(self):
        assert self.env.sample_timestep_for_latency is True

    def test_perturb_observations(self):
        assert self.env.perturb_observations is True

    def test_keep_same_env_episodes(self):
        assert self.env.keep_same_env_for_num_episodes == 1


class TestGateAssetConfig:
    """Test gate asset configuration."""

    def setup_method(self):
        from aerial_gym.config.asset_config.gate_asset_config import gate_asset_params
        self.gate = gate_asset_params

    def test_gate_exists(self):
        assert self.gate is not None

    def test_gate_has_file(self):
        assert hasattr(self.gate, "file")

    def test_gate_num_assets(self):
        assert self.gate.num_assets >= 1
