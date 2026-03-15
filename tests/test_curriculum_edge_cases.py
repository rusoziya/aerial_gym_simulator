"""Edge case tests for curriculum schedules."""
import isaacgym  # noqa: F401
import pytest


@pytest.fixture(scope="module")
def curriculum():
    from aerial_gym.config.task_config.navigation_task_config_gate import task_config
    return task_config.curriculum


@pytest.mark.parametrize("level", [0, 1, 2, -1])
def test_below_min_obstacles_nonneg(curriculum, level):
    count = curriculum.get_obstacle_count_behind_gate(level)
    assert isinstance(count, int) and count >= 0

@pytest.mark.parametrize("level", [24, 30, 50])
def test_above_max_obstacles_valid(curriculum, level):
    count = curriculum.get_obstacle_count_behind_gate(level)
    assert isinstance(count, int) and count >= 0

@pytest.mark.parametrize("level", [0, 1, 2])
def test_below_min_noise_nonneg(curriculum, level):
    g, d = curriculum.get_camera_noise(level)
    assert g >= 0 and d >= 0

@pytest.mark.parametrize("level", [24, 50])
def test_above_max_noise_valid(curriculum, level):
    g, d = curriculum.get_camera_noise(level)
    assert g >= 0

@pytest.mark.parametrize("level", [24, 50])
def test_above_max_spawn_valid(curriculum, level):
    sr = curriculum.get_spawn_ranges(level)
    assert sr["x_half_span_m"] >= 0
