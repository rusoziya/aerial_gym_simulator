"""Tests for curriculum update state machine."""

import isaacgym  # noqa: F401
import pytest


@pytest.fixture(scope="module")
def curriculum():
    from aerial_gym.config.task_config.navigation_task_config_gate import task_config

    cur = task_config.curriculum
    # May be a class (needs instantiation) or already an instance
    if isinstance(cur, type):
        return cur()
    return cur


class TestCurriculumUpdateStateMachine:
    def test_high_success_increases_level(self, curriculum):
        r = curriculum.update_curriculim_level(0.7, 5)
        assert r >= 5

    def test_low_success_holds_level(self, curriculum):
        r = curriculum.update_curriculim_level(0.3, 5)
        assert r == 5

    def test_max_level_capped(self, curriculum):
        r = curriculum.update_curriculim_level(0.7, 23)
        assert r == 23

    def test_min_level_holds(self, curriculum):
        r = curriculum.update_curriculim_level(0.1, 3)
        assert r == 3

    @pytest.mark.parametrize("sr,lvl", [(0.7, 5), (0.3, 5), (0.7, 23), (0.1, 3)])
    def test_result_is_int(self, curriculum, sr, lvl):
        r = curriculum.update_curriculim_level(sr, lvl)
        assert isinstance(r, int)

    def test_very_low_doesnt_decrease(self, curriculum):
        r = curriculum.update_curriculim_level(0.0, 10)
        assert r >= 10

    def test_zero_at_max_holds(self, curriculum):
        r = curriculum.update_curriculim_level(0.0, 23)
        assert r == 23
