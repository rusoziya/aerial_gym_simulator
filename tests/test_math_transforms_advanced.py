"""Advanced behavior tests for transform and quaternion functions."""
import isaacgym  # noqa: F401
import torch
import pytest
from aerial_gym.utils.math import *


class TestTfApply:
    def test_exact_output(self):
        q = torch.tensor([[0, 0, 0.3827, 0.9239]])
        t = torch.tensor([[1.0, 2.0, 3.0]])
        v = torch.tensor([[1.0, 0.0, 0.0]])
        r = tf_apply(q, t, v)
        assert r[0, 0].item() == pytest.approx(1.7071, abs=1e-3)
        assert r[0, 1].item() == pytest.approx(2.7072, abs=1e-3)
        assert r[0, 2].item() == pytest.approx(3.0, abs=1e-3)

    def test_identity_transform(self):
        q = torch.tensor([[0, 0, 0, 1.0]])
        t = torch.zeros(1, 3)
        v = torch.tensor([[5.0, 3.0, 1.0]])
        r = tf_apply(q, t, v)
        assert torch.allclose(r, v, atol=1e-4)


class TestTfInverse:
    def test_exact_output(self):
        q = torch.tensor([[0, 0, 0.3827, 0.9239]])
        t = torch.tensor([[1.0, 2.0, 3.0]])
        qi, ti = tf_inverse(q, t)
        assert qi[0, 2].item() == pytest.approx(-0.3827, abs=1e-3)
        assert ti[0, 0].item() == pytest.approx(-2.1214, abs=1e-2)

    def test_roundtrip(self):
        q = torch.tensor([[0, 0, 0.3827, 0.9239]])
        t = torch.tensor([[1.0, 2.0, 3.0]])
        qi, ti = tf_inverse(q, t)
        q2, t2 = tf_inverse(qi, ti)
        assert torch.allclose(q2, q, atol=1e-3)
        assert torch.allclose(t2, t, atol=1e-2)


class TestTfCombine:
    def test_identity_left(self):
        q1 = torch.tensor([[0, 0, 0, 1.0]])
        t1 = torch.zeros(1, 3)
        q2 = torch.tensor([[0, 0, 0.3827, 0.9239]])
        t2 = torch.tensor([[1.0, 2.0, 3.0]])
        qr, tr = tf_combine(q1, t1, q2, t2)
        assert torch.allclose(qr, q2, atol=1e-3)
        assert torch.allclose(tr, t2, atol=1e-3)

    def test_with_inverse_gives_identity(self):
        q = torch.tensor([[0, 0, 0.3827, 0.9239]])
        t = torch.tensor([[1.0, 2.0, 3.0]])
        qi, ti = tf_inverse(q, t)
        qr, tr = tf_combine(q, t, qi, ti)
        assert abs(qr[0, 3].item()) == pytest.approx(1.0, abs=1e-2)
        assert torch.norm(tr).item() < 0.1


class TestQuatFromAngleAxis:
    def test_90_deg_z(self):
        q = quat_from_angle_axis(torch.tensor([1.5708]), torch.tensor([[0.0, 0.0, 1.0]]))
        assert q[0, 2].item() == pytest.approx(0.7071, abs=1e-3)
        assert q[0, 3].item() == pytest.approx(0.7071, abs=1e-3)

    def test_zero_angle_identity(self):
        q = quat_from_angle_axis(torch.tensor([0.0]), torch.tensor([[0.0, 0.0, 1.0]]))
        assert q[0, 3].item() == pytest.approx(1.0, abs=1e-4)


class TestQuatUnit:
    def test_normalizes(self):
        q = quat_unit(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        assert torch.norm(q).item() == pytest.approx(1.0, abs=1e-4)

    def test_exact_values(self):
        q = quat_unit(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        assert q[0, 0].item() == pytest.approx(0.1826, abs=1e-3)
        assert q[0, 3].item() == pytest.approx(0.7303, abs=1e-3)

    def test_already_unit(self):
        q = torch.tensor([[0, 0, 0, 1.0]])
        assert torch.allclose(quat_unit(q), q, atol=1e-5)


class TestNormalizeAngle:
    @pytest.mark.parametrize("inp,expected", [
        (0.0, 0.0), (3.5, -2.7832), (-3.5, 2.7832), (7.0, 0.7168), (-7.0, -0.7168),
    ])
    def test_exact_values(self, inp, expected):
        r = normalize_angle(torch.tensor([inp]))
        assert r.item() == pytest.approx(expected, abs=1e-3)

    def test_output_in_range(self):
        angles = torch.linspace(-10, 10, 100)
        r = normalize_angle(angles)
        assert all(-torch.pi - 0.01 <= v <= torch.pi + 0.01 for v in r)


class TestComputeVeeMap:
    def test_extracts_vector(self):
        skew = torch.tensor([[[0, -3, 2], [3, 0, -1], [-2, 1, 0.0]]])
        v = compute_vee_map(skew)
        assert v[0, 0].item() == pytest.approx(1.0, abs=1e-4)
        assert v[0, 1].item() == pytest.approx(2.0, abs=1e-4)
        assert v[0, 2].item() == pytest.approx(3.0, abs=1e-4)

    def test_zero_skew(self):
        skew = torch.zeros(1, 3, 3)
        v = compute_vee_map(skew)
        assert torch.allclose(v, torch.zeros(1, 3), atol=1e-5)


class TestGetBasisVector:
    def test_identity_x(self):
        q = torch.tensor([[0, 0, 0, 1.0]])
        r = get_basis_vector(q, torch.tensor([[1, 0, 0.0]]))
        assert torch.allclose(r, torch.tensor([[1, 0, 0.0]]), atol=1e-4)

    def test_identity_z(self):
        q = torch.tensor([[0, 0, 0, 1.0]])
        r = get_basis_vector(q, torch.tensor([[0, 0, 1.0]]))
        assert torch.allclose(r, torch.tensor([[0, 0, 1.0]]), atol=1e-4)


class TestQuatApply:
    def test_identity_passthrough(self):
        q = torch.tensor([[0, 0, 0, 1.0]])
        v = torch.tensor([[1.0, 2.0, 3.0]])
        r = quat_apply(q, v)
        assert torch.allclose(r, v, atol=1e-4)

    def test_inverse_roundtrip(self):
        q = torch.tensor([[0, 0, 0.3827, 0.9239]])
        v = torch.tensor([[1.0, 2.0, 3.0]])
        rotated = quat_apply(q, v)
        back = quat_apply_inverse(q, rotated)
        assert torch.allclose(back, v, atol=1e-3)
