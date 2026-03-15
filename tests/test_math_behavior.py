"""Exact numerical behavior tests for math utility functions.

Locks down precise outputs for specific inputs to detect any formula changes.
"""
import isaacgym  # noqa: F401
import torch
import pytest


class TestQuatRotateExactOutputs:
    def setup_method(self):
        from aerial_gym.utils.math import quat_rotate, quat_rotate_inverse
        self.rot = quat_rotate
        self.inv = quat_rotate_inverse

    def test_identity_exact(self):
        r = self.rot(torch.tensor([[0, 0, 0, 1.0]]), torch.tensor([[1.0, 0.0, 0.0]]))
        assert r[0, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert r[0, 1].item() == pytest.approx(0.0, abs=1e-5)
        assert r[0, 2].item() == pytest.approx(0.0, abs=1e-5)

    def test_90z_exact(self):
        q = torch.tensor([[0, 0, 0.7071068, 0.7071068]])
        v = torch.tensor([[1.0, 0.0, 0.0]])
        r = self.rot(q, v)
        assert r[0, 0].item() == pytest.approx(0.0, abs=1e-4)
        assert r[0, 1].item() == pytest.approx(1.0, abs=1e-4)
        assert r[0, 2].item() == pytest.approx(0.0, abs=1e-4)

    def test_45y_exact(self):
        q = torch.tensor([[0, 0.3826834, 0, 0.9238795]])
        v = torch.tensor([[1.0, 0.0, 0.0]])
        r = self.rot(q, v)
        assert r[0, 0].item() == pytest.approx(0.7071, abs=1e-3)
        assert r[0, 2].item() == pytest.approx(-0.7071, abs=1e-3)

    def test_inverse_90z_exact(self):
        q = torch.tensor([[0, 0, 0.7071068, 0.7071068]])
        v = torch.tensor([[1.0, 0.0, 0.0]])
        r = self.inv(q, v)
        assert r[0, 0].item() == pytest.approx(0.0, abs=1e-4)
        assert r[0, 1].item() == pytest.approx(-1.0, abs=1e-4)


class TestEulerConversionExactOutputs:
    def setup_method(self):
        from aerial_gym.utils.math import get_euler_xyz_tensor
        self.euler = get_euler_xyz_tensor

    def test_identity_quaternion(self):
        q = torch.tensor([[0, 0, 0, 1.0]])
        e = self.euler(q)
        assert all(abs(e[0, i].item()) < 1e-4 for i in range(3))

    def test_90z_quaternion(self):
        q = torch.tensor([[0, 0, 0.7071068, 0.7071068]])
        e = self.euler(q)
        assert e[0, 2].item() == pytest.approx(1.5708, abs=1e-3)

    def test_45y_quaternion(self):
        q = torch.tensor([[0, 0.3826834, 0, 0.9238795]])
        e = self.euler(q)
        assert e[0, 1].item() == pytest.approx(0.7854, abs=1e-3)


class TestSSAExactOutputs:
    def setup_method(self):
        from aerial_gym.utils.math import ssa
        self.ssa = ssa

    @pytest.mark.parametrize("input_val,expected", [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (7.0, 0.7168),
    ])
    def test_exact_values(self, input_val, expected):
        r = self.ssa(torch.tensor([input_val]))
        assert r.item() == pytest.approx(expected, abs=1e-3)


class TestRewardFunctionExactOutputs:
    def setup_method(self):
        from aerial_gym.utils.math import exponential_reward_function, exponential_penalty_function
        self.reward = exponential_reward_function
        self.penalty = exponential_penalty_function

    @pytest.mark.parametrize("val,expected_reward,expected_penalty", [
        (0.0, 1.0, 0.0),
        (0.5, 0.77880079, -0.22119921),
        (1.0, 0.36787945, -0.63212055),
        (2.0, 0.01831564, -0.98168439),
    ])
    def test_exact_values_bw1(self, val, expected_reward, expected_penalty):
        r = self.reward(1.0, 1.0, torch.tensor([val]))
        p = self.penalty(1.0, 1.0, torch.tensor([val]))
        assert r.item() == pytest.approx(expected_reward, abs=1e-6)
        assert p.item() == pytest.approx(expected_penalty, abs=1e-6)
