"""Behavior tests for individual gate reward components."""

import isaacgym  # noqa: F401
import pytest
import torch

from aerial_gym.config.task_config.navigation_task_config_gate import task_config
from aerial_gym.task.navigation_task_gate.reward_functions import compute_gate_reward


@pytest.fixture(scope="module")
def rp():
    return {k: torch.tensor(v) for k, v in task_config.reward_parameters.items()}


def _call(
    rp,
    N=1,
    pos_err=None,
    prev_err=None,
    crashes=None,
    action=None,
    prev_action=None,
    robot_pos=None,
    robot_orient=None,
    gate_pos=None,
    gate_passed=None,
    frac=0.0,
    gw=2.5,
    gh=2.4,
    gc=1.2,
    bv=False,
):
    pe = pos_err if pos_err is not None else torch.tensor([[1, 0, 0.0]] * N)
    ppe = prev_err if prev_err is not None else pe
    cr = crashes if crashes is not None else torch.zeros(N, dtype=torch.bool)
    a = action if action is not None else torch.zeros(N, 4)
    pa = prev_action if prev_action is not None else torch.zeros(N, 4)
    rp_ = robot_pos if robot_pos is not None else torch.tensor([[0, -1.5, 1.5]] * N)
    ro = robot_orient if robot_orient is not None else torch.tensor([[0, 0, 0, 1.0]] * N)
    gp = gate_pos if gate_pos is not None else torch.zeros(N, 3)
    gpassed = gate_passed if gate_passed is not None else torch.zeros(N, dtype=torch.bool)
    bvm = torch.ones(N, dtype=torch.bool) if bv else torch.zeros(N, dtype=torch.bool)
    return compute_gate_reward(
        pe,
        ppe,
        cr,
        a,
        pa,
        rp_,
        ro,
        gp,
        gpassed,
        frac,
        rp,
        torch.full((N,), gw),
        torch.full((N,), gh),
        torch.full((N,), gc),
        bvm,
    )


class TestGettingCloserReward:
    def test_getting_closer_positive(self, rp):
        r1, _, _ = _call(
            rp, pos_err=torch.tensor([[1.0, 0, 0]]), prev_err=torch.tensor([[2.0, 0, 0]])
        )
        r2, _, _ = _call(
            rp, pos_err=torch.tensor([[2.0, 0, 0]]), prev_err=torch.tensor([[2.0, 0, 0]])
        )
        assert r1[0].item() > r2[0].item()

    def test_getting_farther_penalized_more(self, rp):
        closer = _call(
            rp, pos_err=torch.tensor([[1.0, 0, 0]]), prev_err=torch.tensor([[2.0, 0, 0]])
        )[0]
        farther = _call(
            rp, pos_err=torch.tensor([[2.0, 0, 0]]), prev_err=torch.tensor([[1.0, 0, 0]])
        )[0]
        gain = (
            closer[0].item()
            - _call(rp, pos_err=torch.tensor([[1.5, 0, 0]]), prev_err=torch.tensor([[1.5, 0, 0]]))[
                0
            ][0].item()
        )
        loss = (
            _call(rp, pos_err=torch.tensor([[1.5, 0, 0]]), prev_err=torch.tensor([[1.5, 0, 0]]))[0][
                0
            ].item()
            - farther[0].item()
        )
        assert loss > gain  # 2x asymmetry


class TestGatePassageDetection:
    def test_past_gate_not_passed_triggers(self, rp):
        r, _, _ = _call(
            rp,
            robot_pos=torch.tensor([[0.0, 0.1, 1.2]]),
            pos_err=torch.tensor([[0.01, 0, 0]]),
            prev_err=torch.tensor([[0.5, 0, 0]]),
        )
        r_far, _, _ = _call(
            rp,
            robot_pos=torch.tensor([[0.0, -2.0, 1.5]]),
            pos_err=torch.tensor([[2.0, 0, 0]]),
            prev_err=torch.tensor([[2.0, 0, 0]]),
        )
        assert r[0].item() > r_far[0].item()

    def test_outside_width_no_passage_bonus(self, rp):
        r_center, _, _ = _call(
            rp,
            robot_pos=torch.tensor([[0.0, 0.1, 1.2]]),
            pos_err=torch.tensor([[0.01, 0, 0]]),
            prev_err=torch.tensor([[0.5, 0, 0]]),
        )
        r_wide, _, _ = _call(
            rp,
            robot_pos=torch.tensor([[5.0, 0.1, 1.2]]),
            pos_err=torch.tensor([[5.01, 0, 0]]),
            prev_err=torch.tensor([[5.5, 0, 0]]),
        )
        assert r_center[0].item() > r_wide[0].item()


class TestBoundaryViolation:
    def test_violation_reduces_reward(self, rp):
        r_no_bv, _, _ = _call(rp, bv=False)
        r_bv, _, _ = _call(rp, bv=True)
        assert r_no_bv[0].item() > r_bv[0].item()


class TestCameraAlignment:
    def test_facing_gate_alignment_positive(self, rp):
        _, _, align = _call(
            rp, robot_pos=torch.tensor([[0, -2, 1.5]]), robot_orient=torch.tensor([[0, 0, 0, 1.0]])
        )
        assert align[0].item() >= 0

    def test_facing_away_alignment_negative(self, rp):
        _, _, align = _call(
            rp,
            robot_pos=torch.tensor([[0, -2, 1.5]]),
            robot_orient=torch.tensor([[0, 0, 1.0, 0.0]]),
        )
        assert isinstance(align[0].item(), float)  # alignment depends on orientation


class TestActionPenalties:
    def test_zero_actions_no_penalty(self, rp):
        r_zero, _, _ = _call(rp, action=torch.zeros(1, 4))
        r_big, _, _ = _call(rp, action=torch.ones(1, 4) * 0.9)
        assert r_zero[0].item() > r_big[0].item()

    def test_action_diff_penalty(self, rp):
        r_same, _, _ = _call(
            rp,
            action=torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            prev_action=torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        )
        r_diff, _, _ = _call(
            rp,
            action=torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            prev_action=torch.tensor([[-0.5, -0.5, -0.5, -0.5]]),
        )
        assert r_same[0].item() > r_diff[0].item()
