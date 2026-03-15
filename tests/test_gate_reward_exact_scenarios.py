"""Exact reward snapshots for specific gate navigation scenarios."""
import isaacgym  # noqa: F401
import torch
import pytest
from aerial_gym.task.navigation_task_gate.navigation_task_gate import compute_gate_reward
from aerial_gym.config.task_config.navigation_task_config_gate import task_config


@pytest.fixture(scope="module")
def rp():
    return {k: torch.tensor(v) for k, v in task_config.reward_parameters.items()}


def _r(rp, pe, ppe, rpos, frac=0.0, crashes=False, action=None, gp=False):
    N = 1
    a = action if action is not None else torch.zeros(N, 4)
    r, _, al = compute_gate_reward(
        torch.tensor([pe], dtype=torch.float32), torch.tensor([ppe], dtype=torch.float32),
        torch.tensor([crashes]), a, torch.zeros(N, 4),
        torch.tensor([rpos], dtype=torch.float32), torch.tensor([[0,0,0,1.0]]),
        torch.zeros(N, 3), torch.tensor([gp]),
        frac, rp,
        torch.full((N,), 2.5), torch.full((N,), 2.4),
        torch.full((N,), 1.2), torch.zeros(N, dtype=torch.bool))
    return r[0].item(), al[0].item()


class TestScenarioSnapshots:
    def test_drone_behind_gate_approaching(self, rp):
        r, _ = _r(rp, [1,0,0], [1.5,0,0], [0,-2,1.5])
        assert r > 0

    def test_drone_at_gate_center(self, rp):
        r, _ = _r(rp, [0.01,0,0], [0.5,0,0], [0,0.01,1.2])
        assert r > 100  # gate passage bonus

    def test_drone_far_away(self, rp):
        r, _ = _r(rp, [10,0,0], [10,0,0], [5,-5,3])
        assert r < 2

    def test_collision_overrides_everything(self, rp):
        r, _ = _r(rp, [0.01,0,0], [0.5,0,0], [0,0.01,1.2], crashes=True)
        assert r == pytest.approx(-100.0)

    def test_high_curriculum_fraction_increases_reward(self, rp):
        r0, _ = _r(rp, [1,0,0], [1.5,0,0], [0,-2,1.5], frac=0.0)
        r1, _ = _r(rp, [1,0,0], [1.5,0,0], [0,-2,1.5], frac=1.0)
        assert r1 > r0

    def test_big_actions_reduce_reward(self, rp):
        r_zero, _ = _r(rp, [1,0,0], [1,0,0], [0,-1.5,1.5])
        r_big, _ = _r(rp, [1,0,0], [1,0,0], [0,-1.5,1.5], action=torch.ones(1,4)*0.9)
        assert r_zero > r_big

    def test_already_passed_gate_lower_reward(self, rp):
        r_fresh, _ = _r(rp, [0.01,0,0], [0.5,0,0], [0,0.01,1.2], gp=False)
        r_passed, _ = _r(rp, [0.01,0,0], [0.5,0,0], [0,0.01,1.2], gp=True)
        assert r_fresh > r_passed

    def test_outside_gate_width_no_passage(self, rp):
        r_inside, _ = _r(rp, [0.01,0,0], [0.5,0,0], [0,0.01,1.2])
        r_outside, _ = _r(rp, [0.01,0,0], [0.5,0,0], [3,0.01,1.2])
        assert r_inside > r_outside
