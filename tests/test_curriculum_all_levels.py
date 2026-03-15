"""Parametrized tests covering ALL curriculum levels 3-23."""
import isaacgym  # noqa: F401
import pytest


@pytest.fixture(scope="module")
def curriculum():
    from aerial_gym.config.task_config.navigation_task_config_gate import task_config
    return task_config.curriculum


@pytest.mark.parametrize("level", list(range(3, 24)))
def test_obstacle_count_valid(curriculum, level):
    count = curriculum.get_obstacle_count_behind_gate(level)
    assert isinstance(count, int)
    assert 0 <= count <= 30


@pytest.mark.parametrize("level", list(range(3, 24)))
def test_camera_noise_valid(curriculum, level):
    gaussian, dropout = curriculum.get_camera_noise(level)
    assert gaussian >= 0
    assert dropout >= 0
    assert gaussian <= 1.0
    assert dropout <= 1.0


@pytest.mark.parametrize("level", list(range(3, 24)))
def test_spawn_ranges_valid(curriculum, level):
    sr = curriculum.get_spawn_ranges(level)
    assert sr["x_half_span_m"] >= 0
    assert sr["z_half_span_m"] >= 0
    assert sr["y_half_span_m"] >= 0


@pytest.mark.parametrize("level", list(range(3, 24)))
def test_state_noise_valid(curriculum, level):
    sn = curriculum.get_state_noise(level)
    assert sn["drone_pos_std_m"] >= 0
    assert sn["static_pos_std_m"] >= 0


@pytest.mark.parametrize("level", list(range(3, 24)))
def test_frame_dropout_valid(curriculum, level):
    fd = curriculum.get_camera_frame_dropout(level)
    assert 0 <= fd["drone_total"] <= 1.0
    assert 0 <= fd["static_total"] <= 1.0


@pytest.mark.parametrize("level", list(range(3, 24)))
def test_camera_difficulty_valid(curriculum, level):
    angle, h_off, d_off = curriculum.get_static_camera_difficulty(level)
    assert angle >= 0
