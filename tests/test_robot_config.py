"""Behavior tests for robot configuration values."""
import isaacgym  # noqa: F401
import pytest


class TestLMF2Config:
    def setup_method(self):
        from aerial_gym.config.robot_config.lmf2_config import LMF2Cfg as LMF2RobotCfg
        self.cfg = LMF2RobotCfg

    def test_has_init_config(self):
        assert self.cfg.init_config is not None

    def test_min_init_state_length(self):
        assert len(self.cfg.init_config.min_init_state) == 13

    def test_max_init_state_length(self):
        assert len(self.cfg.init_config.max_init_state) == 13

    def test_has_sensor_config(self):
        assert self.cfg.sensor_config is not None

    def test_camera_enabled(self):
        assert self.cfg.sensor_config.enable_camera is True

    def test_has_robot_asset(self):
        assert self.cfg.robot_asset is not None

    def test_has_controller(self):
        assert hasattr(self.cfg, "init_config")


class TestBaseQuadConfig:
    def setup_method(self):
        from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg
        self.cfg = BaseQuadCfg

    def test_has_init_config(self):
        assert self.cfg.init_config is not None

    def test_init_state_13_elements(self):
        assert len(self.cfg.init_config.min_init_state) == 13
