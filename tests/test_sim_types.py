"""Tests for backend-agnostic simulation types."""

from __future__ import annotations

import importlib.util
import sys
from unittest.mock import MagicMock

# Stub isaacgym if not available
if "isaacgym" not in sys.modules:
    fake = MagicMock()
    sys.modules["isaacgym"] = fake
    sys.modules["isaacgym.gymapi"] = fake.gymapi
    sys.modules["isaacgym.gymtorch"] = fake.gymtorch

# Load sim_types directly
_spec = importlib.util.spec_from_file_location(
    "aerial_gym.utils.sim_types", "aerial_gym/utils/sim_types.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

Vec3 = _mod.Vec3
Quat = _mod.Quat
Transform = _mod.Transform
CameraProps = _mod.CameraProps


class TestVec3:
    def test_defaults(self) -> None:
        v = Vec3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_construction(self) -> None:
        v = Vec3(1.0, -2.5, 3.7)
        assert v.x == 1.0
        assert v.y == -2.5
        assert v.z == 3.7

    def test_to_tuple(self) -> None:
        v = Vec3(1.0, 2.0, 3.0)
        assert v.to_tuple() == (1.0, 2.0, 3.0)

    def test_frozen(self) -> None:
        v = Vec3(1.0, 2.0, 3.0)
        try:
            v.x = 99.0
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_equality(self) -> None:
        assert Vec3(1.0, 2.0, 3.0) == Vec3(1.0, 2.0, 3.0)
        assert Vec3(1.0, 2.0, 3.0) != Vec3(1.0, 2.0, 4.0)

    def test_hashable(self) -> None:
        s = {Vec3(1.0, 2.0, 3.0), Vec3(1.0, 2.0, 3.0)}
        assert len(s) == 1


class TestQuat:
    def test_defaults_identity(self) -> None:
        q = Quat()
        assert q.x == 0.0 and q.y == 0.0 and q.z == 0.0 and q.w == 1.0

    def test_to_tuple(self) -> None:
        q = Quat(0.1, 0.2, 0.3, 0.9)
        assert q.to_tuple() == (0.1, 0.2, 0.3, 0.9)


class TestTransform:
    def test_defaults(self) -> None:
        t = Transform()
        assert t.p == Vec3()
        assert t.r == Quat()

    def test_custom(self) -> None:
        t = Transform(p=Vec3(1.0, 2.0, 3.0), r=Quat(0, 0, 0, 1))
        assert t.p.x == 1.0
        assert t.r.w == 1.0


class TestCameraProps:
    def test_defaults(self) -> None:
        cp = CameraProps()
        assert cp.width == 240
        assert cp.height == 135
        assert cp.horizontal_fov == 87.0
        assert cp.near_plane == 0.4
        assert cp.far_plane == 20.0
        assert cp.enable_tensors is True

    def test_custom(self) -> None:
        cp = CameraProps(width=1280, height=720, horizontal_fov=90.0)
        assert cp.width == 1280
        assert cp.height == 720

    def test_mutable(self) -> None:
        cp = CameraProps()
        cp.width = 640
        assert cp.width == 640
