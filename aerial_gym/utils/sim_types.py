"""Backend-agnostic simulation types replacing gymapi.Vec3, Transform, etc.

These simple dataclasses replace Isaac Gym-specific types so sensor,
camera, and viewer code doesn't depend on the physics backend. Each
backend converts these to/from its native types at the boundary.

Usage:
    from aerial_gym.utils.sim_types import Vec3, CameraProps

    pos = Vec3(0.0, -3.0, 1.5)
    target = Vec3(0.0, 0.0, 1.5)
    props = CameraProps(width=240, height=135, horizontal_fov=87.0)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vec3:
    """3D vector — replaces gymapi.Vec3."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class Quat:
    """Quaternion (x, y, z, w) — replaces gymapi.Quat."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)


@dataclass(frozen=True)
class Transform:
    """Position + rotation — replaces gymapi.Transform."""

    p: Vec3 = Vec3()
    r: Quat = Quat()


@dataclass
class CameraProps:
    """Camera sensor properties — replaces gymapi.CameraProperties."""

    width: int = 240
    height: int = 135
    horizontal_fov: float = 87.0
    near_plane: float = 0.4
    far_plane: float = 20.0
    enable_tensors: bool = True


def vec3_to_gymapi(v: Vec3) -> object:
    """Convert Vec3 to gymapi.Vec3 (Isaac Gym backend only)."""
    from isaacgym import gymapi

    return gymapi.Vec3(v.x, v.y, v.z)


def camera_props_to_gymapi(props: CameraProps) -> object:
    """Convert CameraProps to gymapi.CameraProperties (Isaac Gym backend only)."""
    from isaacgym import gymapi

    cam = gymapi.CameraProperties()
    cam.width = props.width
    cam.height = props.height
    cam.horizontal_fov = props.horizontal_fov
    cam.near_plane = props.near_plane
    cam.far_plane = props.far_plane
    cam.enable_tensors = props.enable_tensors
    return cam
