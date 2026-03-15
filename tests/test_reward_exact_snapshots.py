"""Exact numerical snapshot tests for reward at many input combinations.
These parametrized tests lock down reward values across a grid of inputs.
"""
import isaacgym  # noqa: F401
import torch
import pytest
from aerial_gym.utils.math import exponential_reward_function, exponential_penalty_function


@pytest.mark.parametrize("mag,bw,val,expected", [
    (1.0, 1.0, 0.0, 1.0),
    (1.0, 1.0, 0.5, 0.7788),
    (1.0, 1.0, 1.0, 0.3679),
    (1.0, 1.0, 2.0, 0.0183),
    (2.0, 1.0, 1.0, 0.7358),
    (1.0, 2.0, 1.0, 0.6065),
    (1.0, 0.5, 1.0, 0.1353),
    (5.0, 3.0, 0.5, 4.6002),
    (0.5, 0.15, 1.0, 0.0006),
    (0.75, 1.25, 1.0, 0.3370),
])
def test_reward_exact_grid(mag, bw, val, expected):
    r = exponential_reward_function(mag, bw, torch.tensor([val]))
    assert r.item() == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("mag,bw,val,expected", [
    (1.0, 1.0, 0.0, 0.0),
    (1.0, 1.0, 0.5, -0.2212),
    (1.0, 1.0, 1.0, -0.6321),
    (1.0, 1.0, 2.0, -0.9817),
    (2.0, 1.0, 1.0, -1.2642),
    (0.8, 3.333, 0.5, -0.0578),
    (0.5, 2.0, 1.0, -0.1967),
])
def test_penalty_exact_grid(mag, bw, val, expected):
    r = exponential_penalty_function(mag, bw, torch.tensor([val]))
    assert r.item() == pytest.approx(expected, abs=1e-3)
