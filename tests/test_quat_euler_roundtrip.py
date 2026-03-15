"""Roundtrip tests: euler -> quat -> euler must preserve angles."""
import isaacgym  # noqa: F401
import torch
import pytest
from aerial_gym.utils.math import quat_from_euler_xyz_tensor, get_euler_xyz_tensor, ssa


@pytest.mark.parametrize("roll,pitch,yaw", [
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
    (0.3, 0.4, 0.5),
    (-0.3, 0.2, -0.7),
    (1.0, 0.5, 1.5),
    (0.0, 0.0, 3.0),
    (-1.0, -0.3, 2.0),
])
def test_euler_quat_roundtrip(roll, pitch, yaw):
    euler = torch.tensor([[roll, pitch, yaw]])
    q = quat_from_euler_xyz_tensor(euler)
    back = get_euler_xyz_tensor(q)
    diff = ssa(back - euler)
    assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-3)
