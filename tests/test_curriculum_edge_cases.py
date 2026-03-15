"""Edge case tests for curriculum schedule functions."""
import isaacgym  # noqa: F401
import torch
import pytest


@pytest.fixture(scope="module")
def curriculum():
    from aerial_gym.config.task_config.navigation_task_config_gate import task_config
    return task_config.curriculum


class TestBelowMinLevel:
    @pytest.mark.parametrize("level", [0, 1, 2, -1, -100])
    def test_below_min_level_obstacles(self, curriculum, level):
        count = curriculum.get_obstacle_count_behind_gate(level)
        assert isinstance(count, int)
        assert count >= 0

    @pytest.mark.parametrize("level", [0, 1, 2])
    def test_below_min_level_noise(self, curriculum, level):
        g, d = curriculum.get_camera_noise(level)
        assert g >= 0
        assert d >= 0


class TestAboveMaxLevel:
    @pytest.mark.parametrize("level", [24, 30, 50, 100])
    def test_above_max_obstacles(self, curriculum, level):
        count = curriculum.get_obstacle_count_behind_gate(level)
        assert isinstance(count, int)
        assert count >= 0

    @pytest.mark.parametrize("level", [24, 50])
    def test_above_max_noise(self, curriculum, level):
        g, d = curriculum.get_camera_noise(level)
        assert g >= 0

    @pytest.mark.parametrize("level", [24, 50])
    def test_above_max_spawn(self, curriculum, level):
        sr = curriculum.get_spawn_ranges(level)
        assert sr["x_half_span_m"] >= 0


class TestCurriculumUpdateEdgeCases:
    def setup_method(self):
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        self.cur = task_config.curriculum()

    def test_zero_success_rate(self):
        r = self.cur.update_curriculim_level(0.0, 10)
        assert r >= 10

    def test_one_success_rate(self):
        r = self.cur.update_curriculim_level(1.0, 10)
        assert r >= 10

    def test_negative_success_rate(self):
        r = self.cur.update_curriculim_level(-0.5, 10)
        assert isinstance(r, int)

    def test_success_rate_above_one(self):
        r = self.cur.update_curriculim_level(1.5, 10)
        assert isinstance(r, int)
