"""Integration tests for TensorState with realistic tensor operations — no Isaac Gym required.

Uses importlib to load TensorState directly, bypassing aerial_gym.__init__.
"""

from __future__ import annotations

import importlib.util
import sys
from unittest.mock import MagicMock

import pytest

if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["torch"] = MagicMock()

import torch

# Check if torch is real or mocked
_HAS_TORCH = hasattr(torch, "zeros") and not isinstance(torch.zeros, MagicMock)
requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not available (mocked)")

_spec = importlib.util.spec_from_file_location(
    "aerial_gym.env_manager.tensor_state", "aerial_gym/env_manager/tensor_state.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TensorState = _mod.TensorState


@requires_torch
class TestTensorStateRealisticTensorOps:
    """Test TensorState with actual torch tensors mimicking simulation data."""

    def test_store_and_retrieve_position_tensor(self) -> None:
        ts = TensorState()
        num_envs = 64
        pos = torch.zeros(num_envs, 3)
        ts.robot_position = pos
        assert ts.robot_position.shape == (num_envs, 3)
        assert ts["robot_position"] is pos

    def test_in_place_modification_propagates(self) -> None:
        ts = TensorState()
        pos = torch.zeros(8, 3)
        ts.robot_position = pos
        pos[0, 0] = 42.0
        assert ts.robot_position[0, 0].item() == 42.0

    def test_multiple_tensors_independent(self) -> None:
        ts = TensorState()
        ts.robot_position = torch.ones(4, 3)
        ts.robot_orientation = torch.zeros(4, 4)
        assert ts.robot_position.sum().item() == 12.0
        assert ts.robot_orientation.sum().item() == 0.0

    def test_overwrite_tensor_replaces_reference(self) -> None:
        ts = TensorState()
        old = torch.zeros(4, 3)
        ts.robot_position = old
        new = torch.ones(4, 3)
        ts.robot_position = new
        assert ts.robot_position is new
        assert ts.robot_position is not old

    def test_tensor_slice_assignment(self) -> None:
        ts = TensorState()
        ts.robot_actions = torch.zeros(16, 4)
        ts.robot_actions[:, 0] = 1.0
        assert ts.robot_actions[0, 0].item() == 1.0
        assert ts.robot_actions[0, 1].item() == 0.0

    def test_depth_image_tensor_shape(self) -> None:
        ts = TensorState()
        num_envs, num_sensors, h, w = 32, 2, 64, 64
        ts.depth_range_pixels = torch.rand(num_envs, num_sensors, h, w)
        assert ts.depth_range_pixels.shape == (num_envs, num_sensors, h, w)

    def test_segmentation_int32_dtype(self) -> None:
        ts = TensorState()
        seg = torch.zeros(8, 1, 32, 32, dtype=torch.int32)
        ts.segmentation_pixels = seg
        assert ts.segmentation_pixels.dtype == torch.int32

    def test_force_tensor_realistic_shape(self) -> None:
        ts = TensorState()
        total_rigid_bodies = 512
        ts.global_force_tensor = torch.zeros(total_rigid_bodies, 3)
        assert ts.global_force_tensor.shape == (total_rigid_bodies, 3)


@requires_torch
class TestGetSlashSeparatedKeys:
    """Test that .get() works for slash-separated keys (dynamic keys)."""

    def test_get_slash_key_returns_value(self) -> None:
        ts = TensorState()
        ts["sensor/depth/left"] = torch.tensor([1.0, 2.0])
        result = ts.get("sensor/depth/left")
        assert result is not None
        assert result[0].item() == 1.0

    def test_get_slash_key_returns_default_when_missing(self) -> None:
        ts = TensorState()
        result = ts.get("sensor/depth/nonexistent", "fallback")
        assert result == "fallback"

    def test_getitem_slash_key(self) -> None:
        ts = TensorState()
        val = torch.tensor(42.0)
        ts["camera/rgb/front"] = val
        assert ts["camera/rgb/front"] is val

    def test_contains_slash_key_when_set(self) -> None:
        ts = TensorState()
        ts["sensor/imu/accel"] = torch.tensor([0.0, 0.0, 9.81])
        assert "sensor/imu/accel" in ts

    def test_contains_slash_key_when_missing(self) -> None:
        ts = TensorState()
        assert "sensor/imu/accel" not in ts

    def test_multiple_slash_keys_coexist(self) -> None:
        ts = TensorState()
        ts["sensor/depth/left"] = torch.tensor(1.0)
        ts["sensor/depth/right"] = torch.tensor(2.0)
        assert ts.get("sensor/depth/left").item() == 1.0
        assert ts.get("sensor/depth/right").item() == 2.0


class TestContainsEdgeCases:
    """Test __contains__ semantics for all edge cases."""

    def test_none_value_not_contained(self) -> None:
        ts = TensorState()
        assert "robot_position" not in ts

    def test_explicit_none_assignment_not_contained(self) -> None:
        ts = TensorState()
        ts["custom_key"] = "value"
        assert "custom_key" in ts
        ts["custom_key"] = None
        assert "custom_key" not in ts

    def test_zero_int_is_contained(self) -> None:
        ts = TensorState()
        ts["count"] = 0
        assert "count" in ts

    def test_zero_float_is_contained(self) -> None:
        ts = TensorState()
        ts["rate"] = 0.0
        assert "rate" in ts

    def test_false_is_contained(self) -> None:
        ts = TensorState()
        ts["flag"] = False
        assert "flag" in ts

    def test_empty_string_is_contained(self) -> None:
        ts = TensorState()
        ts["label"] = ""
        assert "label" in ts

    def test_empty_list_is_contained(self) -> None:
        ts = TensorState()
        ts["items"] = []
        assert "items" in ts

    def test_empty_tensor_is_contained(self) -> None:
        ts = TensorState()
        ts["empty_t"] = torch.tensor([])
        assert "empty_t" in ts

    def test_nonexistent_key_not_contained(self) -> None:
        ts = TensorState()
        assert "nonexistent_xyz_key" not in ts

    def test_non_string_key_not_contained(self) -> None:
        ts = TensorState()
        assert 42 not in ts  # type: ignore[operator]
        assert None not in ts  # type: ignore[operator]

    def test_builtin_attributes_with_none_not_contained(self) -> None:
        ts = TensorState()
        none_attrs = [
            "robot_position",
            "robot_orientation",
            "crashes",
            "terminations",
            "truncations",
            "depth_range_pixels",
            "segmentation_pixels",
        ]
        for attr in none_attrs:
            assert attr not in ts, f"{attr} should not be 'in' ts when value is None"

    def test_builtin_attrs_with_defaults_contained(self) -> None:
        ts = TensorState()
        assert "dt" in ts  # default 0.0 is not None
        assert "num_robot_actions" in ts  # default 0 is not None
        assert "dof_control_mode" in ts  # default "none" is not None

    def test_pop_removes_from_contains(self) -> None:
        ts = TensorState()
        ts["temp"] = torch.tensor(1.0)
        assert "temp" in ts
        ts.pop("temp")
        assert "temp" not in ts

    def test_update_adds_to_contains(self) -> None:
        ts = TensorState()
        assert "robot_position" not in ts
        ts.update({"robot_position": torch.zeros(4, 3)})
        assert "robot_position" in ts
