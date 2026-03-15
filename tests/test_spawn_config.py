"""Behavior tests for spawn configuration and ranges."""

import isaacgym  # noqa: F401
import pytest


@pytest.fixture(scope="module")
def curriculum():
    from aerial_gym.config.task_config.navigation_task_config_gate import task_config

    return task_config.curriculum


class TestSpawnRangeProperties:
    @pytest.mark.parametrize("level", range(3, 24))
    def test_y_center_constant(self, curriculum, level):
        sr = curriculum.get_spawn_ranges(level)
        assert sr["y_center_m"] == pytest.approx(-1.5, abs=0.5)

    @pytest.mark.parametrize("level", range(3, 24))
    def test_z_center_positive(self, curriculum, level):
        sr = curriculum.get_spawn_ranges(level)
        assert sr["z_center_m"] > 0

    @pytest.mark.parametrize("level", range(3, 24))
    def test_yaw_nonnegative(self, curriculum, level):
        sr = curriculum.get_spawn_ranges(level)
        assert sr["yaw_abs_rad"] >= 0

    def test_x_span_widens_monotonically(self, curriculum):
        prev = 0
        for lvl in range(3, 24):
            span = curriculum.get_spawn_ranges(lvl)["x_half_span_m"]
            assert span >= prev - 0.01
            prev = span

    def test_z_span_widens(self, curriculum):
        s3 = curriculum.get_spawn_ranges(3)["z_half_span_m"]
        s23 = curriculum.get_spawn_ranges(23)["z_half_span_m"]
        assert s23 >= s3
