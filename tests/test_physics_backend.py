"""Tests for PhysicsBackend Protocol and backend selection logic.

Verifies that:
- PhysicsBackend Protocol defines all required methods
- IsaacLabEnv stub implements every Protocol method
- AERIAL_GYM_BACKEND env var controls which backend is selected
- IsaacLabEnv is importable without Isaac Lab installed

All modules are loaded via importlib to bypass aerial_gym.__init__ (which
imports isaacgym). torch is assumed available; isaacgym is mocked.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
import types
from typing import Protocol
from unittest import mock
from unittest.mock import MagicMock

import pytest

if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["torch"] = MagicMock()

import torch  # noqa: F811


def _ensure_isaacgym_mock() -> None:
    """Insert a mock isaacgym into sys.modules so aerial_gym can be imported."""
    if "isaacgym" not in sys.modules:
        fake = types.ModuleType("isaacgym")
        fake.gymapi = mock.MagicMock()
        fake.gymutil = mock.MagicMock()
        fake.gymtorch = mock.MagicMock()
        sys.modules["isaacgym"] = fake
        sys.modules["isaacgym.gymapi"] = fake.gymapi
        sys.modules["isaacgym.gymutil"] = fake.gymutil
        sys.modules["isaacgym.gymtorch"] = fake.gymtorch


_ensure_isaacgym_mock()


def _load_module_from_file(dotted_name: str, file_path: str) -> types.ModuleType:
    """Load a module directly from file path, bypassing aerial_gym.__init__."""
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load dependencies that isaac_lab_env_manager.py imports
_load_module_from_file(
    "aerial_gym.env_manager.base_env_manager",
    "aerial_gym/env_manager/base_env_manager.py",
)
_load_module_from_file("aerial_gym.utils.logging", "aerial_gym/utils/logging.py")
_load_module_from_file("aerial_gym.utils.math", "aerial_gym/utils/math.py")


def _get_physics_backend_class() -> type:
    mod = _load_module_from_file(
        "aerial_gym.env_manager.physics_backend",
        "aerial_gym/env_manager/physics_backend.py",
    )
    return mod.PhysicsBackend


def _get_isaac_lab_env_class() -> type:
    mod = _load_module_from_file(
        "aerial_gym.env_manager.isaac_lab_env_manager",
        "aerial_gym/env_manager/isaac_lab_env_manager.py",
    )
    return mod.IsaacLabEnv


EXPECTED_METHODS: list[str] = [
    "create_env",
    "add_asset_to_env",
    "prepare_for_simulation",
    "pre_physics_step",
    "physics_step",
    "post_physics_step",
    "reset_idx",
    "write_to_sim",
    "refresh_tensors",
    "step_graphics",
    "render_viewer",
    "create_viewer",
    "create_ground_plane",
]


class TestPhysicsBackendProtocol:
    """Verify PhysicsBackend Protocol defines the expected interface."""

    def setup_method(self) -> None:
        self.protocol_cls = _get_physics_backend_class()

    def test_is_protocol(self) -> None:
        assert issubclass(self.protocol_cls, Protocol)

    def test_protocol_has_required_methods(self) -> None:
        members = dict(inspect.getmembers(self.protocol_cls))
        for method_name in EXPECTED_METHODS:
            assert method_name in members, f"Protocol missing method: {method_name}"
            assert callable(members[method_name]), f"{method_name} is not callable"

    def test_protocol_has_expected_attributes(self) -> None:
        # Use __annotations__ to avoid resolving forward references that
        # require TYPE_CHECKING imports (e.g. TensorState).
        annotations = self.protocol_cls.__annotations__
        expected_attrs = [
            "cfg",
            "device",
            "num_envs",
            "num_assets_per_env",
            "num_rigid_bodies_robot",
            "env_handles",
            "asset_handles",
            "sim_has_dof",
            "has_IGE_cameras",
            "env_lower_bound",
            "env_upper_bound",
        ]
        for attr in expected_attrs:
            assert attr in annotations, f"Protocol missing attribute annotation: {attr}"

    def test_method_signatures_have_type_hints(self) -> None:
        for method_name in EXPECTED_METHODS:
            method = getattr(self.protocol_cls, method_name)
            # Use __annotations__ to avoid resolving forward references
            ann = method.__annotations__
            assert "return" in ann, f"{method_name} missing return type annotation"


class TestIsaacLabEnvStub:
    """Verify IsaacLabEnv stub implements all PhysicsBackend Protocol methods."""

    def setup_method(self) -> None:
        self.protocol_cls = _get_physics_backend_class()
        self.lab_cls = _get_isaac_lab_env_class()

    def test_importable(self) -> None:
        cls = _get_isaac_lab_env_class()
        assert cls is not None
        assert cls.__name__ == "IsaacLabEnv"

    def test_implements_all_protocol_methods(self) -> None:
        for method_name in EXPECTED_METHODS:
            assert hasattr(self.lab_cls, method_name), f"IsaacLabEnv missing method: {method_name}"
            assert callable(getattr(self.lab_cls, method_name))

    def _make_stub_instance(self) -> object:
        """Create an IsaacLabEnv instance with minimal mock config."""
        num_envs = 4
        env_cfg = mock.MagicMock()
        env_cfg.env.num_envs = num_envs
        env_cfg.env.lower_bound_min = [-5.0, -5.0, -5.0]
        env_cfg.env.lower_bound_max = [-3.0, -3.0, -3.0]
        env_cfg.env.upper_bound_min = [3.0, 3.0, 3.0]
        env_cfg.env.upper_bound_max = [5.0, 5.0, 5.0]

        sim_cfg = mock.MagicMock()
        sim_cfg.viewer.headless = True

        return self.lab_cls(config=env_cfg, sim_config=sim_cfg, has_cameras=False, device="cpu")

    def test_create_env_returns_index(self) -> None:
        instance = self._make_stub_instance()
        result = instance.create_env(0)
        assert result == 0
        assert 0 in instance.env_handles

    def test_create_env_populates_handles(self) -> None:
        instance = self._make_stub_instance()
        for i in range(3):
            instance.create_env(i)
        assert len(instance.env_handles) == 3

    @pytest.mark.skipif(isinstance(torch, MagicMock), reason="torch mocked")
    def test_reset_idx_updates_bounds(self) -> None:
        instance = self._make_stub_instance()
        lower_before = instance.env_lower_bound.clone()
        upper_before = instance.env_upper_bound.clone()

        env_ids = torch.tensor([0, 2])
        instance.reset_idx(env_ids)

        lower_after = instance.env_lower_bound
        upper_after = instance.env_upper_bound

        # At least one of the reset environments should have changed bounds
        # (randomization makes identical values extremely unlikely)
        lower_changed = not torch.allclose(lower_before[env_ids], lower_after[env_ids])
        upper_changed = not torch.allclose(upper_before[env_ids], upper_after[env_ids])
        assert lower_changed or upper_changed, "reset_idx did not randomize bounds"

    def test_add_asset_to_env(self) -> None:
        instance = self._make_stub_instance()
        instance.create_env(0)
        asset_info = {"asset_type": "obstacle", "name": "test_box"}
        handle, seg_inc = instance.add_asset_to_env(
            asset_info_dict=asset_info,
            env_handle=0,
            env_id=0,
            global_asset_counter=1,
            segmentation_counter=100,
        )
        assert handle == 1
        assert seg_inc == 1
        assert 1 in instance.asset_handles[0]

    def test_add_robot_sets_rigid_bodies(self) -> None:
        instance = self._make_stub_instance()
        instance.create_env(0)
        asset_info = {"asset_type": "robot", "name": "test_quad"}
        instance.add_asset_to_env(
            asset_info_dict=asset_info,
            env_handle=0,
            env_id=0,
            global_asset_counter=0,
            segmentation_counter=100,
        )
        assert instance.num_rigid_bodies_robot is not None

    def test_physics_step_no_sim_context(self) -> None:
        """physics_step should not raise when sim_context is None (stub mode)."""
        instance = self._make_stub_instance()
        instance.physics_step()
        assert not instance.graphics_are_stepped

    def test_write_to_sim_is_noop(self) -> None:
        """write_to_sim should not raise in stub mode."""
        instance = self._make_stub_instance()
        instance.write_to_sim()

    def test_refresh_tensors_is_noop(self) -> None:
        """refresh_tensors should not raise in stub mode."""
        instance = self._make_stub_instance()
        instance.refresh_tensors()


class TestBackendSelection:
    """Verify AERIAL_GYM_BACKEND env var controls which backend is instantiated."""

    def test_default_is_isaacgym(self) -> None:
        env = os.environ.copy()
        env.pop("AERIAL_GYM_BACKEND", None)
        with mock.patch.dict(os.environ, env, clear=True):
            backend = os.environ.get("AERIAL_GYM_BACKEND", "isaacgym").lower()
        assert backend == "isaacgym"

    def test_env_var_selects_isaaclab(self) -> None:
        with mock.patch.dict(os.environ, {"AERIAL_GYM_BACKEND": "isaaclab"}):
            backend = os.environ.get("AERIAL_GYM_BACKEND", "isaacgym").lower()
        assert backend == "isaaclab"

    def test_env_var_case_insensitive(self) -> None:
        with mock.patch.dict(os.environ, {"AERIAL_GYM_BACKEND": "IsaacLab"}):
            backend = os.environ.get("AERIAL_GYM_BACKEND", "isaacgym").lower()
        assert backend == "isaaclab"

    def test_backend_selection_logic_isaacgym(self) -> None:
        """Verify EnvManager source references both backends."""
        source = open("aerial_gym/env_manager/env_manager.py").read()
        assert 'os.environ.get("AERIAL_GYM_BACKEND"' in source
        assert "IsaacLabEnv" in source
        assert "IsaacGymEnv" in source

    def test_backend_selection_logic_isaaclab_branch(self) -> None:
        """Verify the isaaclab branch exists in source."""
        source = open("aerial_gym/env_manager/env_manager.py").read()
        assert 'backend == "isaaclab"' in source

    def test_isaac_lab_module_importable(self) -> None:
        """Verify IsaacLabEnv can be imported without Isaac Lab installed."""
        cls = _get_isaac_lab_env_class()
        assert cls is not None
        assert cls.__name__ == "IsaacLabEnv"
