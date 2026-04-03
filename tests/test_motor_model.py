"""Behavior tests for motor model JIT functions (gate_config_10)."""

import isaacgym  # noqa: F401
import pytest
import torch


class TestDiscreteMixingFactor:
    def setup_method(self):
        from aerial_gym.control.motor_model import discrete_mixing_factor

        self.fn = discrete_mixing_factor

    def test_exact_value(self):
        r = self.fn(0.01, torch.tensor([0.05]))
        assert r.item() == pytest.approx(16.6667, abs=0.01)

    def test_zero_time_constant(self):
        r = self.fn(0.01, torch.tensor([0.0]))
        assert r.item() == pytest.approx(100.0, abs=0.1)

    def test_batch(self):
        r = self.fn(0.01, torch.tensor([0.05, 0.1, 0.02]))
        assert r.shape == (3,)


class TestMotorModelRate:
    def setup_method(self):
        from aerial_gym.control.motor_model import motor_model_rate

        self.fn = motor_model_rate

    def test_within_limits(self):
        r = self.fn(torch.tensor([1.0]), torch.tensor([10.0]), torch.tensor([100.0]))
        assert r.item() == pytest.approx(10.0, abs=0.1)

    def test_clamped_positive(self):
        r = self.fn(torch.tensor([100.0]), torch.tensor([10.0]), torch.tensor([5.0]))
        assert r.item() == pytest.approx(5.0, abs=0.1)

    def test_clamped_negative(self):
        r = self.fn(torch.tensor([-100.0]), torch.tensor([10.0]), torch.tensor([5.0]))
        assert r.item() == pytest.approx(-5.0, abs=0.1)

    def test_zero_error_zero_rate(self):
        r = self.fn(torch.tensor([0.0]), torch.tensor([10.0]), torch.tensor([100.0]))
        assert r.item() == pytest.approx(0.0, abs=1e-5)


class TestThrustComputation:
    def setup_method(self):
        from aerial_gym.control.motor_model import compute_thrust_with_force_time_constant

        self.fn = compute_thrust_with_force_time_constant

    def test_at_reference_stays(self):
        ref = torch.tensor([5.0])
        cur = torch.tensor([5.0])
        r = self.fn(ref, cur, torch.tensor([10.0]), torch.tensor([100.0]), 0.01)
        assert r.item() == pytest.approx(5.0, abs=0.1)

    def test_approaches_reference(self):
        ref = torch.tensor([10.0])
        cur = torch.tensor([5.0])
        r = self.fn(ref, cur, torch.tensor([10.0]), torch.tensor([100.0]), 0.01)
        assert 5.0 < r.item() < 10.0

    def test_batch(self):
        r = self.fn(torch.ones(4), torch.zeros(4), torch.ones(4) * 10, torch.ones(4) * 100, 0.01)
        assert r.shape == (4,)
        assert all(r[i].item() > 0 for i in range(4))
