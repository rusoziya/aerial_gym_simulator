"""Tests for gate environment configuration."""

import isaacgym  # noqa: F401
import pytest


class TestGateEnvConfig:
    def setup_method(self):
        from aerial_gym.config.env_config.gate_env import GateEnvCfg

        self.cfg = GateEnvCfg

    def test_env_class_exists(self):
        assert self.cfg.env is not None

    def test_collision_force_threshold(self):
        assert self.cfg.env.collision_force_threshold == pytest.approx(0.05, abs=0.01)

    def test_reset_on_collision(self):
        assert self.cfg.env.reset_on_collision is True

    def test_use_warp(self):
        assert self.cfg.env.use_warp is True

    def test_has_env_bounds(self):
        bounds = self.cfg.env
        assert bounds.env_spacing > 0


class TestEnvAssetConfig:
    def setup_method(self):
        from aerial_gym.config.asset_config.env_asset_config import (
            OBJECT_SEMANTIC_ID,
            WALL_SEMANTIC_ID,
            back_wall,
            bottom_wall,
            front_wall,
            left_wall,
            right_wall,
            top_wall,
        )

        self.walls = [left_wall, right_wall, top_wall, bottom_wall, front_wall, back_wall]
        self.WALL_SEMANTIC_ID = WALL_SEMANTIC_ID
        self.OBJECT_SEMANTIC_ID = OBJECT_SEMANTIC_ID

    def test_six_walls_exist(self):
        assert len(self.walls) == 6

    def test_all_walls_have_num_assets_1(self):
        for wall in self.walls:
            assert wall.num_assets == 1

    def test_all_walls_have_state_ratios(self):
        for wall in self.walls:
            assert hasattr(wall, "min_state_ratio")
            assert hasattr(wall, "max_state_ratio")
            assert len(wall.min_state_ratio) == 13
            assert len(wall.max_state_ratio) == 13

    def test_walls_keep_in_env(self):
        for wall in self.walls:
            assert wall.keep_in_env is True

    def test_semantic_ids_defined(self):
        assert self.WALL_SEMANTIC_ID == 8
        assert self.OBJECT_SEMANTIC_ID == 3

        from aerial_gym.control.control_allocation import ControlAllocator

        self.CA = ControlAllocator
