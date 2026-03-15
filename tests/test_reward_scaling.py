"""Parametrized tests for reward function scaling behavior across input ranges."""
import isaacgym  # noqa: F401
import torch
import pytest
from aerial_gym.utils.math import exponential_reward_function, exponential_penalty_function


@pytest.mark.parametrize("magnitude", [0.5, 1.0, 2.5, 5.0, 100.0])
def test_reward_at_zero_equals_magnitude(magnitude):
    r = exponential_reward_function(magnitude, 1.0, torch.tensor([0.0]))
    assert r.item() == pytest.approx(magnitude, abs=1e-5)


@pytest.mark.parametrize("magnitude", [0.5, 1.0, 2.5, 5.0, 100.0])
def test_penalty_at_zero_equals_zero(magnitude):
    r = exponential_penalty_function(magnitude, 1.0, torch.tensor([0.0]))
    assert r.item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("base_width", [0.1, 0.5, 1.0, 5.0, 10.0])
def test_reward_monotonically_decreases(base_width):
    values = torch.linspace(0, 5, 20)
    rewards = exponential_reward_function(1.0, base_width, values)
    for i in range(len(rewards) - 1):
        assert rewards[i].item() >= rewards[i + 1].item()


@pytest.mark.parametrize("base_width", [0.1, 0.5, 1.0, 5.0, 10.0])
def test_penalty_monotonically_decreases(base_width):
    values = torch.linspace(0, 5, 20)
    penalties = exponential_penalty_function(1.0, base_width, values)
    for i in range(len(penalties) - 1):
        assert penalties[i].item() >= penalties[i + 1].item()


@pytest.mark.parametrize("val", [0.0, 0.5, 1.0, 2.0, 5.0])
def test_reward_plus_penalty_equals_magnitude_at_val(val):
    """reward(v) + |penalty(v)| = magnitude for all v."""
    mag = 3.0
    r = exponential_reward_function(mag, 1.0, torch.tensor([val]))
    p = exponential_penalty_function(mag, 1.0, torch.tensor([val]))
    assert (r.item() + abs(p.item())) == pytest.approx(mag, abs=1e-5)
