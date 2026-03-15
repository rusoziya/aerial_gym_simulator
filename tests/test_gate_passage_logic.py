"""Tests for gate passage detection within compute_gate_reward."""
import isaacgym  # noqa: F401
import torch
import pytest
from aerial_gym.task.navigation_task_gate.reward_functions import compute_gate_reward
from aerial_gym.config.task_config.navigation_task_config_gate import task_config


@pytest.fixture(scope="module")
def rp():
    return {k: torch.tensor(v) for k, v in task_config.reward_parameters.items()}


def _reward(rp, robot_y, robot_x=0.0, robot_z=1.2, gate_passed=False):
    N = 1
    r, _, _ = compute_gate_reward(
        torch.tensor([[0.1, 0, 0]]), torch.tensor([[0.5, 0, 0]]),
        torch.zeros(N, dtype=torch.bool), torch.zeros(N, 4), torch.zeros(N, 4),
        torch.tensor([[robot_x, robot_y, robot_z]]),
        torch.tensor([[0, 0, 0, 1.0]]),
        torch.zeros(N, 3), torch.tensor([gate_passed]),
        0.0, rp,
        torch.full((N,), 2.5), torch.full((N,), 2.4),
        torch.full((N,), 1.2), torch.zeros(N, dtype=torch.bool))
    return r[0].item()


class TestGatePassageYPlane:
    def test_behind_gate_no_passage_bonus(self, rp):
        r = _reward(rp, robot_y=-2.0)
        assert r < 200  # no passage bonus

    def test_past_gate_gets_passage_bonus(self, rp):
        r = _reward(rp, robot_y=0.1)
        assert r > 200  # passage bonus kicks in

    def test_well_past_gate_still_passage(self, rp):
        r = _reward(rp, robot_y=2.0)
        assert r > 100


class TestGatePassageWidth:
    def test_centered_passage(self, rp):
        r = _reward(rp, robot_y=0.1, robot_x=0.0)
        assert r > 200

    def test_edge_of_gate_passage(self, rp):
        r_center = _reward(rp, robot_y=0.1, robot_x=0.0)
        r_edge = _reward(rp, robot_y=0.1, robot_x=1.2)
        assert r_center >= r_edge  # center should be >= edge

    def test_outside_gate_width_no_passage(self, rp):
        r_inside = _reward(rp, robot_y=0.1, robot_x=0.0)
        r_outside = _reward(rp, robot_y=0.1, robot_x=3.0)
        assert r_inside > r_outside


class TestGatePassageHeight:
    def test_at_gate_center_height(self, rp):
        r = _reward(rp, robot_y=0.1, robot_z=1.2)
        assert r > 200

    def test_above_gate_no_passage(self, rp):
        r_inside = _reward(rp, robot_y=0.1, robot_z=1.2)
        r_above = _reward(rp, robot_y=0.1, robot_z=5.0)
        assert r_inside > r_above

    def test_below_gate_no_passage(self, rp):
        r_inside = _reward(rp, robot_y=0.1, robot_z=1.2)
        r_below = _reward(rp, robot_y=0.1, robot_z=-1.0)
        assert r_inside > r_below


class TestAlreadyPassedGate:
    def test_no_double_passage_reward(self, rp):
        r_fresh = _reward(rp, robot_y=0.1, gate_passed=False)
        r_passed = _reward(rp, robot_y=0.1, gate_passed=True)
        assert r_fresh > r_passed
