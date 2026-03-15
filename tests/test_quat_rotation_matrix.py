"""Behavior tests for quaternion-to-rotation-matrix conversion."""
import isaacgym  # noqa: F401
import torch
import pytest
from aerial_gym.utils.math import quat_rotate, quat_rotate_inverse


class TestRotationProperties:
    """Test mathematical properties that rotations must satisfy."""

    @pytest.mark.parametrize("angle", [0.0, 0.5, 1.0, 1.5708, 3.14159])
    def test_rotation_preserves_norm(self, angle):
        q = torch.tensor([[0, 0, torch.sin(torch.tensor(angle/2)).item(), torch.cos(torch.tensor(angle/2)).item()]])
        v = torch.tensor([[3.0, 4.0, 5.0]])
        r = quat_rotate(q, v)
        assert torch.norm(r).item() == pytest.approx(torch.norm(v).item(), abs=1e-4)

    @pytest.mark.parametrize("angle", [0.0, 0.5, 1.0, 1.5708, 3.14159])
    def test_inverse_undoes_rotation(self, angle):
        q = torch.tensor([[0, 0, torch.sin(torch.tensor(angle/2)).item(), torch.cos(torch.tensor(angle/2)).item()]])
        v = torch.tensor([[1.0, 2.0, 3.0]])
        assert torch.allclose(quat_rotate_inverse(q, quat_rotate(q, v)), v, atol=1e-4)

    @pytest.mark.parametrize("axis", [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [-1, 0, 0], [0, -1, 0], [0, 0, -1],
    ])
    def test_rotation_around_each_axis(self, axis):
        angle = 1.0
        ax = torch.tensor([axis], dtype=torch.float32)
        ax = ax / torch.norm(ax)
        s = torch.sin(torch.tensor(angle / 2)).item()
        c = torch.cos(torch.tensor(angle / 2)).item()
        q = torch.tensor([[ax[0,0]*s, ax[0,1]*s, ax[0,2]*s, c]])
        v = torch.tensor([[1.0, 1.0, 1.0]])
        r = quat_rotate(q, v)
        assert torch.norm(r).item() == pytest.approx(torch.norm(v).item(), abs=1e-4)

    def test_batch_rotation(self):
        N = 16
        q = torch.tensor([[0, 0, 0, 1.0]]).repeat(N, 1)
        v = torch.randn(N, 3)
        r = quat_rotate(q, v)
        assert torch.allclose(r, v, atol=1e-5)
