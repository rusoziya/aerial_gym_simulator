"""Edge case tests for reward computation."""

import isaacgym  # noqa: F401
import pytest
import torch

from aerial_gym.config.task_config.navigation_task_config_gate import task_config
from aerial_gym.task.navigation_task_gate.reward_functions import compute_gate_reward


@pytest.fixture(scope="module")
def rp():
    return {k: torch.tensor(v) for k, v in task_config.reward_parameters.items()}


def _gate_reward(rp, N=1, **overrides):
    defaults = dict(
        pos_error=torch.ones(N, 3),
        prev_pos_error=torch.ones(N, 3),
        crashes=torch.zeros(N, dtype=torch.bool),
        action=torch.zeros(N, 4),
        prev_action=torch.zeros(N, 4),
        robot_pos=torch.tensor([[0, -1.5, 1.5]] * N),
        robot_orient=torch.tensor([[0, 0, 0, 1.0]] * N),
        gate_pos=torch.zeros(N, 3),
        gate_passed=torch.zeros(N, dtype=torch.bool),
        frac=0.0,
        gw=torch.full((N,), 2.5),
        gh=torch.full((N,), 2.4),
        gc=torch.full((N,), 1.2),
        bv=torch.zeros(N, dtype=torch.bool),
    )
    defaults.update(overrides)
    d = defaults
    return compute_gate_reward(
        d["pos_error"],
        d["prev_pos_error"],
        d["crashes"],
        d["action"],
        d["prev_action"],
        d["robot_pos"],
        d["robot_orient"],
        d["gate_pos"],
        d["gate_passed"],
        d["frac"],
        rp,
        d["gw"],
        d["gh"],
        d["gc"],
        d["bv"],
    )


class TestZeroInputs:
    def test_all_zeros_finite(self, rp):
        r, c, a = _gate_reward(rp, pos_error=torch.zeros(1, 3))
        assert torch.isfinite(r).all()

    def test_all_zeros_no_crash(self, rp):
        _, c, _ = _gate_reward(rp)
        assert not c.any()


class TestLargeInputs:
    def test_very_far_position(self, rp):
        r, _, _ = _gate_reward(rp, pos_error=torch.tensor([[100, 100, 100.0]]))
        assert torch.isfinite(r).all()

    def test_very_large_actions(self, rp):
        r, _, _ = _gate_reward(rp, action=torch.ones(1, 4) * 100)
        assert torch.isfinite(r).all()


class TestBatchSizes:
    @pytest.mark.parametrize("N", [1, 2, 4, 8, 16, 32])
    def test_various_batch_sizes(self, rp, N):
        r, c, a = _gate_reward(rp, N=N)
        assert r.shape == (N,)
        assert c.shape == (N,)
        assert a.shape == (N,)


class TestMixedCrashes:
    def test_partial_crash(self, rp):
        crashes = torch.tensor([True, False, True, False])
        r, _, _ = _gate_reward(rp, N=4, crashes=crashes)
        assert r[0].item() == pytest.approx(-100.0, abs=1)
        assert r[1].item() != -100.0
        assert r[2].item() == pytest.approx(-100.0, abs=1)
        assert r[3].item() != -100.0


class TestGatePassedFlag:
    def test_already_passed_no_double_reward(self, rp):
        r_fresh, _, _ = _gate_reward(
            rp,
            gate_passed=torch.zeros(1, dtype=torch.bool),
            robot_pos=torch.tensor([[0.0, 0.1, 1.2]]),
            pos_error=torch.tensor([[0.01, 0, 0]]),
            prev_pos_error=torch.tensor([[0.5, 0, 0]]),
        )
        r_passed, _, _ = _gate_reward(
            rp,
            gate_passed=torch.ones(1, dtype=torch.bool),
            robot_pos=torch.tensor([[0.0, 0.1, 1.2]]),
            pos_error=torch.tensor([[0.01, 0, 0]]),
            prev_pos_error=torch.tensor([[0.5, 0, 0]]),
        )
        assert r_fresh[0].item() > r_passed[0].item()
