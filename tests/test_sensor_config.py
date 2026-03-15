"""Behavior tests for sensor configuration values."""
import isaacgym  # noqa: F401
import pytest


class TestBaseDepthCameraConfig:
    def setup_method(self):
        from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import BaseDepthCameraConfig
        self.cfg = BaseDepthCameraConfig

    def test_width(self):
        assert self.cfg.width == 240

    def test_height(self):
        assert self.cfg.height == 135

    def test_num_sensors(self):
        assert self.cfg.num_sensors >= 1


class TestBaseLidarConfig:
    def setup_method(self):
        from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import BaseLidarConfig
        self.cfg = BaseLidarConfig

    def test_has_range(self):
        assert hasattr(self.cfg, 'max_range') or hasattr(self.cfg, 'range')

    def test_has_num_sensors(self):
        assert self.cfg.num_sensors >= 1


class TestBaseImuConfig:
    def setup_method(self):
        from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig
        self.cfg = BaseImuConfig

    def test_exists(self):
        assert self.cfg is not None
