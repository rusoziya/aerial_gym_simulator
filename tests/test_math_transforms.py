"""Behavior tests for math transform functions (gate_config_10)."""

import isaacgym  # noqa: F401
import pytest
import torch

from aerial_gym.utils.math import *


class TestTorchInterpolateRatio:
    def test_midpoint(self):
        r = torch_interpolate_ratio(
            torch.tensor([[0.0]]), torch.tensor([[10.0]]), torch.tensor([[0.5]])
        )
        assert r[0, 0].item() == pytest.approx(5.0, abs=1e-5)

    def test_endpoints(self):
        mn, mx = torch.tensor([[0.0, 0.0]]), torch.tensor([[10.0, 20.0]])
        assert torch_interpolate_ratio(mn, mx, torch.tensor([[0.0, 0.0]]))[
            0, 0
        ].item() == pytest.approx(0.0)
        assert torch_interpolate_ratio(mn, mx, torch.tensor([[1.0, 1.0]]))[
            0, 0
        ].item() == pytest.approx(10.0)

    def test_batch(self):
        r = torch_interpolate_ratio(
            torch.zeros(4, 3), torch.ones(4, 3) * 10, torch.ones(4, 3) * 0.5
        )
        assert r.shape == (4, 3)
        assert torch.allclose(r, torch.ones(4, 3) * 5, atol=1e-5)


class TestTensorClamp:
    def test_clamps_below(self):
        assert (
            tensor_clamp(torch.tensor([-5.0]), torch.tensor([-1.0]), torch.tensor([1.0]))[0].item()
            == -1.0
        )

    def test_clamps_above(self):
        assert (
            tensor_clamp(torch.tensor([5.0]), torch.tensor([-1.0]), torch.tensor([1.0]))[0].item()
            == 1.0
        )

    def test_passthrough(self):
        assert (
            tensor_clamp(torch.tensor([0.5]), torch.tensor([-1.0]), torch.tensor([1.0]))[0].item()
            == 0.5
        )

    def test_exact_values(self):
        t = torch.tensor([-2.0, 0.5, 3.0])
        r = tensor_clamp(t, torch.tensor([-1.0, -1.0, -1.0]), torch.tensor([1.0, 1.0, 1.0]))
        assert r.tolist() == [-1.0, 0.5, 1.0]


class TestScaleUnscale:
    def test_scale_endpoints(self):
        r = scale(torch.tensor([-1.0, 0.0, 1.0]), torch.tensor([0.0] * 3), torch.tensor([10.0] * 3))
        assert r[0].item() == pytest.approx(0.0, abs=1e-5)
        assert r[1].item() == pytest.approx(5.0, abs=1e-5)
        assert r[2].item() == pytest.approx(10.0, abs=1e-5)

    def test_unscale_endpoints(self):
        r = unscale(
            torch.tensor([0.0, 5.0, 10.0]), torch.tensor([0.0] * 3), torch.tensor([10.0] * 3)
        )
        assert r[0].item() == pytest.approx(-1.0, abs=1e-5)
        assert r[1].item() == pytest.approx(0.0, abs=1e-5)
        assert r[2].item() == pytest.approx(1.0, abs=1e-5)

    def test_roundtrip(self):
        x = torch.tensor([-0.7, 0.3, 0.9])
        lo, hi = torch.tensor([2.0] * 3), torch.tensor([8.0] * 3)
        assert torch.allclose(unscale(scale(x, lo, hi), lo, hi), x, atol=1e-5)


class TestQuatFromEuler:
    def test_identity(self):
        q = quat_from_euler_xyz_tensor(torch.zeros(1, 3))
        assert torch.allclose(q, torch.tensor([[0, 0, 0, 1.0]]), atol=1e-4)

    def test_90_yaw(self):
        q = quat_from_euler_xyz_tensor(torch.tensor([[0.0, 0.0, 1.5708]]))
        assert q[0, 2].item() == pytest.approx(0.7071, abs=1e-3)
        assert q[0, 3].item() == pytest.approx(0.7071, abs=1e-3)

    def test_45_roll(self):
        q = quat_from_euler_xyz_tensor(torch.tensor([[0.7854, 0.0, 0.0]]))
        assert q[0, 0].item() == pytest.approx(0.3827, abs=1e-3)
        assert q[0, 3].item() == pytest.approx(0.9239, abs=1e-3)

    def test_roundtrip_with_euler(self):
        euler = torch.tensor([[0.3, 0.5, 0.7]])
        q = quat_from_euler_xyz_tensor(euler)
        back = get_euler_xyz_tensor(q)
        assert torch.allclose(ssa(back), ssa(euler), atol=1e-3)


class TestVehicleFrameQuat:
    def test_identity_stays_identity(self):
        q = torch.tensor([[0, 0, 0, 1.0]])
        vf = vehicle_frame_quat_from_quat(q)
        assert torch.allclose(vf, q, atol=1e-4)

    def test_extracts_yaw_only(self):
        q = torch.tensor([[0.1, 0.2, 0.3, 0.9]])
        vf = vehicle_frame_quat_from_quat(q)
        assert vf[0, 0].item() == pytest.approx(0.0, abs=1e-3)
        assert vf[0, 1].item() == pytest.approx(0.0, abs=1e-3)
        assert vf[0, 2].item() == pytest.approx(0.3424, abs=1e-3)


class TestNormalize:
    def test_unit_vector(self):
        r = normalize(torch.tensor([[3.0, 4.0, 0.0]]))
        assert r[0, 0].item() == pytest.approx(0.6, abs=1e-4)
        assert r[0, 1].item() == pytest.approx(0.8, abs=1e-4)
        assert torch.norm(r).item() == pytest.approx(1.0, abs=1e-4)

    def test_already_unit(self):
        v = torch.tensor([[0.0, 1.0, 0.0]])
        assert torch.allclose(normalize(v), v, atol=1e-5)


class TestPDControl:
    def test_proportional_only(self):
        r = pd_control(
            torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([10.0]), torch.tensor([0.0])
        )
        assert r[0].item() == pytest.approx(10.0, abs=1e-4)

    def test_derivative_only(self):
        r = pd_control(
            torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([5.0])
        )
        assert r[0].item() == pytest.approx(5.0, abs=1e-4)

    def test_combined(self):
        r = pd_control(
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.1, 0.2]),
            torch.tensor([10.0, 10.0]),
            torch.tensor([1.0, 1.0]),
        )
        assert r[0].item() == pytest.approx(10.1, abs=1e-3)
        assert r[1].item() == pytest.approx(20.2, abs=1e-3)


class TestQuatMul:
    def test_identity_left(self):
        q_id = torch.tensor([[0, 0, 0, 1.0]])
        q = torch.tensor([[0.1, 0.2, 0.3, 0.9]])
        r = quat_mul(q_id, q)
        assert torch.allclose(r, q, atol=1e-4)

    def test_identity_right(self):
        q_id = torch.tensor([[0, 0, 0, 1.0]])
        q = torch.tensor([[0.1, 0.2, 0.3, 0.9]])
        r = quat_mul(q, q_id)
        assert torch.allclose(r, q, atol=1e-4)

    def test_inverse_gives_identity(self):
        q = torch.tensor([[0, 0, 0.3827, 0.9239]])
        q_inv = quat_conjugate(q)
        r = quat_mul(q, q_inv)
        assert r[0, 3].abs().item() == pytest.approx(1.0, abs=1e-3)


class TestQuatConjugate:
    def test_negates_xyz(self):
        q = torch.tensor([[0.1, 0.2, 0.3, 0.9]])
        c = quat_conjugate(q)
        assert c[0, 0].item() == pytest.approx(-0.1, abs=1e-5)
        assert c[0, 1].item() == pytest.approx(-0.2, abs=1e-5)
        assert c[0, 2].item() == pytest.approx(-0.3, abs=1e-5)
        assert c[0, 3].item() == pytest.approx(0.9, abs=1e-5)

    def test_double_conjugate_identity(self):
        q = torch.tensor([[0.1, 0.2, 0.3, 0.9]])
        assert torch.allclose(quat_conjugate(quat_conjugate(q)), q, atol=1e-5)
