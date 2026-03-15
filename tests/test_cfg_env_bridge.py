"""Tests for cfg_env_bridge — no Isaac Gym or Sample Factory required.

Uses importlib to load cfg_env_bridge directly, stubbing out sample_factory
to avoid importing the full framework.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

# Stub out sample_factory.utils.typing so cfg_env_bridge can import Config
_sf_typing_stub = ModuleType("sample_factory.utils.typing")
_sf_typing_stub.Config = SimpleNamespace  # type: ignore[attr-defined]
sys.modules.setdefault("sample_factory", ModuleType("sample_factory"))
sys.modules.setdefault("sample_factory.utils", ModuleType("sample_factory.utils"))
sys.modules.setdefault("sample_factory.utils.typing", _sf_typing_stub)

_spec = importlib.util.spec_from_file_location(
    "aerial_gym.rl_training.sample_factory.aerialgym_examples.cfg_env_bridge",
    "aerial_gym/rl_training/sample_factory/aerialgym_examples/cfg_env_bridge.py",
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

bridge_cfg_to_env_vars: type = _mod.bridge_cfg_to_env_vars  # type: ignore[assignment]


def _make_cfg(**overrides: object) -> SimpleNamespace:
    """Build a SimpleNamespace that mirrors the CLI args registered by train_aerialgym_custom_net_gate."""
    defaults: dict[str, object] = {
        "fusion": "gated",
        "gate_per_feature": 1,
        "disable_static_camera_orientation_randomization": False,
        "disable_camera_noise_randomization": False,
        "disable_camera_frame_dropout_randomization": False,
        "disable_state_noise_randomization": False,
        "disable_dynamic_camera_following": False,
        "enable_static_camera_yaw_sweep": False,
        "enable_static_camera_locked": False,
        "disable_dynamic_follow_gate_blending": False,
        "disable_spawn_position_randomization": False,
        "disable_spawn_orientation_randomization": False,
        "disable_curriculum_multiplier": False,
        "enable_static_camera_arc_follow": False,
        "static_camera_yaw_sweep_speed_deg": 10.0,
        "disable_drone_camera_noise_randomization": None,
        "disable_static_camera_noise_randomization": None,
        "disable_drone_camera_frame_dropout": None,
        "disable_static_camera_frame_dropout": None,
        "static_camera_base_y": None,
        "static_camera_base_z": None,
        "static_camera_arc_radius_m": 2.0,
        "dynamic_camera_follow_y_offset_m": None,
        "enable_dynamic_camera_following": None,
        "force_curriculum_level": None,
        "evaluation": False,
        "min_curriculum_level": None,
        "max_curriculum_level": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


_ENV_KEYS_SET_BY_DEFAULT: list[str] = [
    "SF_FUSION_MODE",
    "SF_GATE_PER_FEATURE",
    "SF_DISABLE_STATIC_CAMERA_ORIENT_RANDOMIZATION",
    "SF_DISABLE_CAMERA_NOISE_RANDOMIZATION",
    "SF_DISABLE_CAMERA_FRAME_DROPOUT_RANDOMIZATION",
    "SF_DISABLE_STATE_NOISE_RANDOMIZATION",
    "disable_dynamic_camera_following",
    "SF_ENABLE_STATIC_CAMERA_YAW_SWEEP",
    "SF_STATIC_CAMERA_LOCKED_FOLLOW",
    "SF_DISABLE_DYNAMIC_FOLLOW_GATE_BLENDING",
    "SF_DISABLE_SPAWN_POSITION_RANDOMIZATION",
    "SF_DISABLE_SPAWN_ORIENTATION_RANDOMIZATION",
    "SF_DISABLE_CURRICULUM_MULTIPLIER",
    "SF_ENABLE_STATIC_CAMERA_ARC_FOLLOW",
    "SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG",
    "SF_STATIC_CAMERA_ARC_RADIUS_M",
]

_ALL_OPTIONAL_KEYS: list[str] = [
    "SF_FORCE_CURRICULUM_LEVEL",
    "SF_MIN_CURRICULUM_LEVEL",
    "SF_MAX_CURRICULUM_LEVEL",
    "SF_DISABLE_DRONE_CAMERA_NOISE_RANDOMIZATION",
    "SF_DISABLE_STATIC_CAMERA_NOISE_RANDOMIZATION",
    "SF_DISABLE_DRONE_CAMERA_FRAME_DROPOUT",
    "SF_DISABLE_STATIC_CAMERA_FRAME_DROPOUT",
    "SF_STATIC_CAMERA_BASE_Y",
    "SF_STATIC_CAMERA_BASE_Z",
    "SF_DYNAMIC_CAMERA_FOLLOW_OFFSET_Y",
    "enable_dynamic_camera_following",
]


@pytest.fixture(autouse=True)
def _clean_env() -> None:
    """Remove all bridge env vars before and after each test."""
    all_keys = _ENV_KEYS_SET_BY_DEFAULT + _ALL_OPTIONAL_KEYS
    for key in all_keys:
        os.environ.pop(key, None)
    yield  # type: ignore[misc]
    for key in all_keys:
        os.environ.pop(key, None)


class TestBasicEnvVars:
    def test_fusion_mode_set(self) -> None:
        cfg = _make_cfg(fusion="concat")
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_FUSION_MODE"] == "concat"

    def test_gate_per_feature_true(self) -> None:
        cfg = _make_cfg(gate_per_feature=1)
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_GATE_PER_FEATURE"] == "1"

    def test_gate_per_feature_false(self) -> None:
        cfg = _make_cfg(gate_per_feature=0)
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_GATE_PER_FEATURE"] == "0"

    def test_all_default_keys_are_set(self) -> None:
        cfg = _make_cfg()
        bridge_cfg_to_env_vars(cfg)
        for key in _ENV_KEYS_SET_BY_DEFAULT:
            assert key in os.environ, f"Expected env var {key} to be set"

    def test_bool_flags_false_by_default(self) -> None:
        cfg = _make_cfg()
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_DISABLE_CAMERA_NOISE_RANDOMIZATION"] == "false"
        assert os.environ["SF_DISABLE_STATE_NOISE_RANDOMIZATION"] == "false"

    def test_bool_flags_true_when_enabled(self) -> None:
        cfg = _make_cfg(
            disable_camera_noise_randomization=True,
            disable_state_noise_randomization=True,
        )
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_DISABLE_CAMERA_NOISE_RANDOMIZATION"] == "true"
        assert os.environ["SF_DISABLE_STATE_NOISE_RANDOMIZATION"] == "true"

    def test_yaw_sweep_speed(self) -> None:
        cfg = _make_cfg(static_camera_yaw_sweep_speed_deg=25.5)
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG"] == "25.5"


class TestPerCameraOverrides:
    def test_none_overrides_do_not_set_env(self) -> None:
        cfg = _make_cfg()
        bridge_cfg_to_env_vars(cfg)
        assert "SF_DISABLE_DRONE_CAMERA_NOISE_RANDOMIZATION" not in os.environ
        assert "SF_DISABLE_STATIC_CAMERA_NOISE_RANDOMIZATION" not in os.environ
        assert "SF_DISABLE_DRONE_CAMERA_FRAME_DROPOUT" not in os.environ
        assert "SF_DISABLE_STATIC_CAMERA_FRAME_DROPOUT" not in os.environ

    def test_explicit_overrides_set_env(self) -> None:
        cfg = _make_cfg(
            disable_drone_camera_noise_randomization=True,
            disable_static_camera_noise_randomization=False,
            disable_drone_camera_frame_dropout=True,
            disable_static_camera_frame_dropout=False,
        )
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_DISABLE_DRONE_CAMERA_NOISE_RANDOMIZATION"] == "true"
        assert os.environ["SF_DISABLE_STATIC_CAMERA_NOISE_RANDOMIZATION"] == "false"
        assert os.environ["SF_DISABLE_DRONE_CAMERA_FRAME_DROPOUT"] == "true"
        assert os.environ["SF_DISABLE_STATIC_CAMERA_FRAME_DROPOUT"] == "false"


class TestCurriculumFlags:
    def test_force_curriculum_level_int(self) -> None:
        cfg = _make_cfg(force_curriculum_level=13)
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_FORCE_CURRICULUM_LEVEL"] == "13"

    def test_force_curriculum_level_string(self) -> None:
        cfg = _make_cfg(force_curriculum_level="7")
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_FORCE_CURRICULUM_LEVEL"] == "7"

    def test_force_curriculum_level_none_does_not_set(self) -> None:
        cfg = _make_cfg(force_curriculum_level=None)
        bridge_cfg_to_env_vars(cfg)
        assert "SF_FORCE_CURRICULUM_LEVEL" not in os.environ

    def test_force_curriculum_level_string_none_clears(self) -> None:
        os.environ["SF_FORCE_CURRICULUM_LEVEL"] = "5"
        cfg = _make_cfg(force_curriculum_level="none")
        bridge_cfg_to_env_vars(cfg)
        assert "SF_FORCE_CURRICULUM_LEVEL" not in os.environ

    def test_min_curriculum_level_set(self) -> None:
        cfg = _make_cfg(min_curriculum_level=3)
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_MIN_CURRICULUM_LEVEL"] == "3"

    def test_max_curriculum_level_set(self) -> None:
        cfg = _make_cfg(min_curriculum_level=2, max_curriculum_level=10)
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_MIN_CURRICULUM_LEVEL"] == "2"
        assert os.environ["SF_MAX_CURRICULUM_LEVEL"] == "10"

    def test_max_curriculum_without_min_not_set(self) -> None:
        cfg = _make_cfg(min_curriculum_level=None, max_curriculum_level=10)
        bridge_cfg_to_env_vars(cfg)
        assert "SF_MIN_CURRICULUM_LEVEL" not in os.environ
        assert "SF_MAX_CURRICULUM_LEVEL" not in os.environ

    def test_evaluation_mode_skips_min_max(self) -> None:
        cfg = _make_cfg(evaluation=True, min_curriculum_level=5, max_curriculum_level=10)
        bridge_cfg_to_env_vars(cfg)
        assert "SF_MIN_CURRICULUM_LEVEL" not in os.environ
        assert "SF_MAX_CURRICULUM_LEVEL" not in os.environ


class TestStaticCameraPosition:
    def test_base_y_float(self) -> None:
        cfg = _make_cfg(static_camera_base_y=-3.5)
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_STATIC_CAMERA_BASE_Y"] == "-3.5"

    def test_base_y_none_not_set(self) -> None:
        cfg = _make_cfg(static_camera_base_y=None)
        bridge_cfg_to_env_vars(cfg)
        assert "SF_STATIC_CAMERA_BASE_Y" not in os.environ

    def test_base_z_float(self) -> None:
        cfg = _make_cfg(static_camera_base_z=2.0)
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_STATIC_CAMERA_BASE_Z"] == "2.0"

    def test_base_z_adaptive(self) -> None:
        cfg = _make_cfg(static_camera_base_z="adaptive")
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_STATIC_CAMERA_BASE_Z"] == "adaptive"

    def test_base_z_none_not_set(self) -> None:
        cfg = _make_cfg(static_camera_base_z=None)
        bridge_cfg_to_env_vars(cfg)
        assert "SF_STATIC_CAMERA_BASE_Z" not in os.environ

    def test_arc_radius(self) -> None:
        cfg = _make_cfg(static_camera_arc_radius_m=4.5)
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_STATIC_CAMERA_ARC_RADIUS_M"] == "4.5"


class TestDynamicCameraFollowing:
    def test_enable_dynamic_following_none_not_set(self) -> None:
        cfg = _make_cfg(enable_dynamic_camera_following=None)
        bridge_cfg_to_env_vars(cfg)
        assert "enable_dynamic_camera_following" not in os.environ

    def test_enable_dynamic_following_true(self) -> None:
        cfg = _make_cfg(enable_dynamic_camera_following=True)
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["enable_dynamic_camera_following"] == "true"

    def test_follow_y_offset_set(self) -> None:
        cfg = _make_cfg(dynamic_camera_follow_y_offset_m=1.5)
        bridge_cfg_to_env_vars(cfg)
        assert os.environ["SF_DYNAMIC_CAMERA_FOLLOW_OFFSET_Y"] == "1.5"

    def test_follow_y_offset_none_not_set(self) -> None:
        cfg = _make_cfg(dynamic_camera_follow_y_offset_m=None)
        bridge_cfg_to_env_vars(cfg)
        assert "SF_DYNAMIC_CAMERA_FOLLOW_OFFSET_Y" not in os.environ
