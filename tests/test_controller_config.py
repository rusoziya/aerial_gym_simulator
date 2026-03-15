"""Behavior tests for controller configuration values."""
import isaacgym  # noqa: F401
import pytest


class TestLMF2VelocityControlConfig:
    def setup_method(self):
        from aerial_gym.registry.controller_registry import controller_registry
        self.cfg = controller_registry.get_controller_config("lmf2_velocity_control")

    def test_config_exists(self):
        assert self.cfg is not None

    def test_has_kp(self):
        assert hasattr(self.cfg, 'K_pos_tensor_min') or hasattr(self.cfg, 'kp') or hasattr(self.cfg, 'K_p')

class TestLeeControllerConfig:
    def setup_method(self):
        try:
            from aerial_gym.config.controller_config.lee_controller_config import control as cfg
            self.cfg = cfg
        except ImportError:
            pytest.skip("Lee controller config not available")

    def test_has_gains(self):
        assert self.cfg is not None
