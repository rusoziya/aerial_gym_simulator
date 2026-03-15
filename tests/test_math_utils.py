"""Behavior-capture tests for aerial_gym.utils.math (gate_config_10).
Formula: magnitude * exp(-val^2 / base_width)
"""
import isaacgym  # noqa: F401
import torch
import pytest

class TestExponentialRewardFunction:
    def setup_method(self):
        from aerial_gym.utils.math import exponential_reward_function
        self.fn = exponential_reward_function

    def test_zero_value_gives_full_magnitude(self):
        assert torch.allclose(self.fn(2.0, 1.0, torch.tensor([0.0])), torch.tensor([2.0]), atol=1e-6)

    def test_large_value_near_zero(self):
        assert self.fn(1.0, 1.0, torch.tensor([10.0])).item() < 1e-10

    def test_symmetry(self):
        assert torch.allclose(self.fn(1.0, 1.0, torch.tensor([3.0])), self.fn(1.0, 1.0, torch.tensor([-3.0])), atol=1e-6)

    def test_magnitude_scales(self):
        r1 = self.fn(1.0, 1.0, torch.tensor([1.0]))
        assert torch.allclose(self.fn(2.0, 1.0, torch.tensor([1.0])), 2.0 * r1, atol=1e-6)

    def test_wider_base_width(self):
        assert self.fn(1.0, 5.0, torch.tensor([1.0])).item() > self.fn(1.0, 0.5, torch.tensor([1.0])).item()

    def test_batch(self):
        r = self.fn(1.0, 1.0, torch.tensor([0.0, 1.0, 2.0, 5.0]))
        assert r.shape == (4,)
        assert r[0] > r[1] > r[2] > r[3]

    def test_exact_formula(self):
        mag, bw, val = 3.0, 2.0, 1.5
        expected = mag * torch.exp(torch.tensor(-(val**2) / bw))
        assert torch.allclose(self.fn(mag, bw, torch.tensor([val])), expected, atol=1e-6)

    def test_known_value(self):
        assert self.fn(1.0, 1.0, torch.tensor([1.0])).item() == pytest.approx(0.3678794, abs=1e-5)


class TestExponentialPenaltyFunction:
    def setup_method(self):
        from aerial_gym.utils.math import exponential_penalty_function
        self.fn = exponential_penalty_function

    def test_zero_gives_zero(self):
        assert torch.allclose(self.fn(1.0, 1.0, torch.tensor([0.0])), torch.tensor([0.0]), atol=1e-6)

    def test_nonzero_negative(self):
        assert self.fn(1.0, 1.0, torch.tensor([1.0])).item() < 0

    def test_larger_more_negative(self):
        assert self.fn(1.0, 1.0, torch.tensor([3.0])).item() < self.fn(1.0, 1.0, torch.tensor([1.0])).item()

    def test_exact_formula(self):
        mag, bw, val = 2.0, 1.5, 1.0
        expected = mag * (torch.exp(torch.tensor(-(val**2) / bw)) - 1.0)
        assert torch.allclose(self.fn(mag, bw, torch.tensor([val])), expected, atol=1e-6)

    def test_at_infinity_equals_neg_magnitude(self):
        assert self.fn(5.0, 1.0, torch.tensor([100.0])).item() == pytest.approx(-5.0, abs=1e-5)


class TestSSA:
    def setup_method(self):
        from aerial_gym.utils.math import ssa
        self.ssa = ssa

    def test_zero(self):
        assert abs(self.ssa(torch.tensor([0.0])).item()) < 1e-6

    def test_small_angle(self):
        assert self.ssa(torch.tensor([0.5])).item() == pytest.approx(0.5, abs=1e-5)

    def test_two_pi_wraps(self):
        assert abs(self.ssa(torch.tensor([2 * torch.pi])).item()) < 1e-5

    def test_batch_in_range(self):
        r = self.ssa(torch.tensor([0.0, 0.5, -0.5, 2*torch.pi, -2*torch.pi]))
        for v in r:
            assert -torch.pi - 1e-5 <= v.item() <= torch.pi + 1e-5


class TestQuatRotateInverse:
    def setup_method(self):
        from aerial_gym.utils.math import quat_rotate_inverse, quat_rotate
        self.inv = quat_rotate_inverse
        self.rot = quat_rotate

    def test_identity(self):
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        v = torch.tensor([[1.0, 2.0, 3.0]])
        assert torch.allclose(self.inv(q, v), v, atol=1e-5)

    def test_preserves_magnitude(self):
        a = 1.2
        q = torch.tensor([[0.0, 0.0, torch.sin(torch.tensor(a/2)).item(), torch.cos(torch.tensor(a/2)).item()]])
        v = torch.tensor([[3.0, 4.0, 0.0]])
        assert torch.norm(self.inv(q, v)).item() == pytest.approx(torch.norm(v).item(), abs=1e-4)

    def test_double_inverse_identity(self):
        a = 0.7
        q = torch.tensor([[0.0, 0.0, torch.sin(torch.tensor(a/2)).item(), torch.cos(torch.tensor(a/2)).item()]])
        v = torch.tensor([[1.0, 2.0, 3.0]])
        assert torch.allclose(self.inv(q, self.rot(q, v)), v, atol=1e-4)

    def test_batch_shape(self):
        q = torch.tensor([[0, 0, 0, 1.0]] * 4)
        v = torch.randn(4, 3)
        assert self.inv(q, v).shape == (4, 3)
