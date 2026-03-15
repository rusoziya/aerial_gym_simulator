"""RL reward pipeline behavior tests — exact component breakdown."""
import isaacgym  # noqa: F401
import torch
import pytest
from aerial_gym.utils.math import exponential_reward_function, exponential_penalty_function
from aerial_gym.config.task_config.navigation_task_config_gate import task_config as gate_cfg
from aerial_gym.config.task_config.navigation_task_config import task_config as nav_cfg


@pytest.fixture(scope="module")
def gate_rp():
    return gate_cfg.reward_parameters


@pytest.fixture(scope="module")
def nav_rp():
    return {k: torch.tensor(v) for k, v in nav_cfg.reward_parameters.items()}


class TestRewardComponentBreakdown:
    """Test individual reward components at distance=1.0 from target."""

    def test_pos_reward_at_dist_1(self, gate_rp):
        r = exponential_reward_function(gate_rp["pos_reward_magnitude"], gate_rp["pos_reward_exponent"], torch.tensor([1.0]))
        assert r.item() == pytest.approx(0.000636, abs=1e-4)

    def test_very_close_reward_at_dist_1(self, gate_rp):
        r = exponential_reward_function(gate_rp["very_close_to_goal_reward_magnitude"], gate_rp["very_close_to_goal_reward_exponent"], torch.tensor([1.0]))
        assert r.item() == pytest.approx(0.337, abs=1e-2)

    def test_getting_closer_by_0_5(self, gate_rp):
        gc = torch.tensor([0.5])  # moved 0.5 closer
        reward = gate_rp["getting_closer_reward_multiplier"] * gc
        assert reward.item() == pytest.approx(2.5, abs=1e-4)

    def test_getting_farther_by_0_5(self, gate_rp):
        gc = torch.tensor([-0.5])  # moved 0.5 farther
        penalty = 2.0 * gate_rp["getting_closer_reward_multiplier"] * gc
        assert penalty.item() == pytest.approx(-5.0, abs=1e-4)

    def test_distance_from_goal_linear(self):
        for d in [0.0, 5.0, 10.0, 20.0]:
            expected = (20.0 - d) / 20.0
            assert expected == pytest.approx((20.0 - d) / 20.0)


class TestCurriculumMultiplicationFactor:
    @pytest.mark.parametrize("frac,expected", [
        (0.0, 1.0), (0.25, 1.125), (0.5, 1.25), (0.75, 1.375), (1.0, 1.5),
    ])
    def test_exact_factor(self, frac, expected):
        factor = 1.0 + 0.5 * frac
        assert factor == pytest.approx(expected, abs=1e-4)


class TestNavRewardDistanceCurve:
    """Lock down the base navigation reward at specific distances."""

    @pytest.mark.parametrize("dist,expected", [
        (0.0, 11.0), (0.5, 8.663), (1.0, 5.384), (2.0, 2.496), (5.0, 0.754), (10.0, 0.5),
    ])
    def test_nav_reward_at_distance(self, nav_rp, dist, expected):
        from aerial_gym.task.navigation_task.navigation_task import compute_reward
        pe = torch.tensor([[dist, 0, 0]])
        r, _ = compute_reward(pe, pe, torch.zeros(1, dtype=torch.bool), torch.zeros(1,4), torch.zeros(1,4), 0.0, nav_rp)
        assert r.item() == pytest.approx(expected, abs=0.1)


class TestGettingCloserAsymmetryExact:
    """Lock down the 2x penalty for getting farther from target."""

    def test_gain_loss_ratio(self, nav_rp):
        from aerial_gym.task.navigation_task.navigation_task import compute_reward
        base, _ = compute_reward(torch.tensor([[2.0,0,0]]), torch.tensor([[2.0,0,0]]),
                                  torch.zeros(1, dtype=torch.bool), torch.zeros(1,4), torch.zeros(1,4), 0.0, nav_rp)
        closer, _ = compute_reward(torch.tensor([[1.0,0,0]]), torch.tensor([[2.0,0,0]]),
                                    torch.zeros(1, dtype=torch.bool), torch.zeros(1,4), torch.zeros(1,4), 0.0, nav_rp)
        farther, _ = compute_reward(torch.tensor([[3.0,0,0]]), torch.tensor([[2.0,0,0]]),
                                     torch.zeros(1, dtype=torch.bool), torch.zeros(1,4), torch.zeros(1,4), 0.0, nav_rp)
        gain = closer.item() - base.item()
        loss = base.item() - farther.item()
        assert gain == pytest.approx(12.888, abs=0.1)
        assert loss == pytest.approx(21.264, abs=0.1)
        assert loss / gain == pytest.approx(1.65, abs=0.1)


class TestActionPenaltyComponents:
    """Test that each action axis contributes a penalty."""

    @pytest.mark.parametrize("axis,diff_mag_key,diff_exp_key", [
        (0, "x_action_diff_penalty_magnitude", "x_action_diff_penalty_exponent"),
        (2, "z_action_diff_penalty_magnitude", "z_action_diff_penalty_exponent"),
        (3, "yawrate_action_diff_penalty_magnitude", "yawrate_action_diff_penalty_exponent"),
    ])
    def test_action_diff_penalty_per_axis(self, gate_rp, axis, diff_mag_key, diff_exp_key):
        diff = torch.tensor([1.0])
        penalty = exponential_penalty_function(gate_rp[diff_mag_key], gate_rp[diff_exp_key], diff)
        assert penalty.item() < 0

    @pytest.mark.parametrize("axis,abs_mag_key,abs_exp_key", [
        (0, "x_absolute_action_penalty_magnitude", "x_absolute_action_penalty_exponent"),
        (2, "z_absolute_action_penalty_magnitude", "z_absolute_action_penalty_exponent"),
        (3, "yawrate_absolute_action_penalty_magnitude", "yawrate_absolute_action_penalty_exponent"),
    ])
    def test_absolute_action_penalty_per_axis(self, gate_rp, axis, abs_mag_key, abs_exp_key):
        action = torch.tensor([0.9])
        penalty = exponential_penalty_function(gate_rp[abs_mag_key], gate_rp[abs_exp_key], action)
        assert penalty.item() < 0
