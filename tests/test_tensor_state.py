"""Tests for TensorState dict-compatible wrapper — no torch/isaacgym required."""

from __future__ import annotations

import importlib.util
import sys
from unittest.mock import MagicMock

# Stub torch if not available (system Python may lack it)
if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["torch"] = MagicMock()

# Load TensorState directly
_spec = importlib.util.spec_from_file_location(
    "aerial_gym.env_manager.tensor_state", "aerial_gym/env_manager/tensor_state.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TensorState = _mod.TensorState


class TestTensorStateInit:
    def test_creates_with_none_defaults(self) -> None:
        ts = TensorState()
        assert ts.robot_position is None
        assert ts.crashes is None
        assert ts.dt == 0.0
        assert ts.num_robot_actions == 0

    def test_known_attributes_exist(self) -> None:
        ts = TensorState()
        expected = [
            "robot_position",
            "robot_orientation",
            "crashes",
            "terminations",
            "truncations",
            "global_force_tensor",
            "depth_range_pixels",
            "env_bounds_min",
            "gravity",
            "dt",
            "robot_actions",
            "gate_variant_counter",
            "curriculum_level",
        ]
        for key in expected:
            assert key in ts.__dict__, f"Missing attribute: {key}"

    def test_has_at_least_69_attributes(self) -> None:
        ts = TensorState()
        assert len(ts.__dict__) >= 69


class TestDictStyleGetSet:
    def test_getitem(self) -> None:
        ts = TensorState()
        ts.robot_position = "mock_tensor"
        assert ts["robot_position"] == "mock_tensor"

    def test_setitem(self) -> None:
        ts = TensorState()
        ts["robot_position"] = "mock_tensor"
        assert ts.robot_position == "mock_tensor"

    def test_get_returns_none_for_unset(self) -> None:
        ts = TensorState()
        assert ts.get("robot_position") is None

    def test_get_returns_default(self) -> None:
        ts = TensorState()
        assert ts.get("nonexistent_key", 42) == 42

    def test_get_returns_value_when_set(self) -> None:
        ts = TensorState()
        ts.robot_position = "data"
        assert ts.get("robot_position") == "data"

    def test_getitem_missing_key_raises(self) -> None:
        ts = TensorState()
        try:
            ts["totally_fake_key"]
            assert False, "Should have raised KeyError"
        except KeyError as e:
            assert "totally_fake_key" in str(e)


class TestContains:
    def test_none_is_not_contained(self) -> None:
        ts = TensorState()
        assert "robot_position" not in ts

    def test_set_value_is_contained(self) -> None:
        ts = TensorState()
        ts.robot_position = "data"
        assert "robot_position" in ts

    def test_nonexistent_key_not_contained(self) -> None:
        ts = TensorState()
        assert "totally_fake_key" not in ts

    def test_zero_is_contained(self) -> None:
        ts = TensorState()
        ts["custom_key"] = 0
        assert "custom_key" in ts

    def test_false_is_contained(self) -> None:
        ts = TensorState()
        ts["flag"] = False
        assert "flag" in ts


class TestDictIteration:
    def test_keys_includes_known(self) -> None:
        ts = TensorState()
        keys = list(ts.keys())
        assert "robot_position" in keys
        assert "crashes" in keys

    def test_items_returns_pairs(self) -> None:
        ts = TensorState()
        ts.robot_position = "data"
        found = dict(ts.items())
        assert found["robot_position"] == "data"

    def test_iter_yields_keys(self) -> None:
        ts = TensorState()
        assert "robot_position" in list(ts)


class TestMutation:
    def test_pop_existing(self) -> None:
        ts = TensorState()
        ts["temp"] = 99
        assert ts.pop("temp") == 99
        assert ts.get("temp") is None

    def test_pop_with_default(self) -> None:
        ts = TensorState()
        assert ts.pop("nonexistent", "fallback") == "fallback"

    def test_update_from_dict(self) -> None:
        ts = TensorState()
        ts.update({"robot_position": "data", "dt": 0.01})
        assert ts.robot_position == "data"
        assert ts.dt == 0.01

    def test_setitem_new_dynamic_key(self) -> None:
        ts = TensorState()
        ts["dynamic_key"] = 123
        assert ts["dynamic_key"] == 123


class TestAttributeAndDictEquivalence:
    def test_set_attr_read_dict(self) -> None:
        ts = TensorState()
        ts.robot_position = "tensor"
        assert ts["robot_position"] == "tensor"

    def test_set_dict_read_attr(self) -> None:
        ts = TensorState()
        ts["robot_position"] = "tensor"
        assert ts.robot_position == "tensor"

    def test_same_object_identity(self) -> None:
        ts = TensorState()
        obj = object()
        ts.robot_position = obj
        assert ts["robot_position"] is obj

    def test_scalar_types(self) -> None:
        ts = TensorState()
        ts.num_robot_actions = 4
        assert ts["num_robot_actions"] == 4
        ts["dt"] = 0.005
        assert ts.dt == 0.005

    def test_string_types(self) -> None:
        ts = TensorState()
        ts.dof_control_mode = "velocity"
        assert ts["dof_control_mode"] == "velocity"
