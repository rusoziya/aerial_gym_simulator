"""Tests for RunConfig YAML loading and --set override parsing — no Isaac Gym required.

Uses importlib to load run_config directly, bypassing aerial_gym.__init__
which imports isaacgym (unavailable on non-GPU / wrong Python version).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Load run_config modules via importlib to avoid isaacgym import
# ---------------------------------------------------------------------------
_enums_spec = importlib.util.spec_from_file_location(
    "aerial_gym.config.run_config_enums", "aerial_gym/config/run_config_enums.py"
)
_enums_mod = importlib.util.module_from_spec(_enums_spec)
sys.modules[_enums_spec.name] = _enums_mod
_enums_spec.loader.exec_module(_enums_mod)

_config_spec = importlib.util.spec_from_file_location(
    "aerial_gym.config.run_config", "aerial_gym/config/run_config.py"
)
_config_mod = importlib.util.module_from_spec(_config_spec)
sys.modules[_config_spec.name] = _config_mod
_config_spec.loader.exec_module(_config_mod)

RunConfig = _config_mod.RunConfig
load_config = _config_mod.load_config
load_config_with_overrides = _config_mod.load_config_with_overrides
_coerce_value = _config_mod._coerce_value
_set_nested = _config_mod._set_nested

CONFIGS_DIR = Path("configs")
YAML_FILES = sorted(CONFIGS_DIR.glob("*.yaml"))


class TestYamlConfigsValidate:
    """Every checked-in YAML config must pass Pydantic validation."""

    @pytest.fixture(params=YAML_FILES, ids=[f.stem for f in YAML_FILES])
    def yaml_path(self, request: pytest.FixtureRequest) -> Path:
        return request.param

    def test_config_loads_without_error(self, yaml_path: Path) -> None:
        cfg = load_config(str(yaml_path))
        assert cfg.common.task is not None

    def test_config_mode_is_valid(self, yaml_path: Path) -> None:
        cfg = load_config(str(yaml_path))
        assert cfg.mode.value in {"train", "eval", "play", "inference_suite"}


class TestCoerceValue:
    """Unit tests for the _coerce_value helper."""

    def test_true_lowercase(self) -> None:
        assert _coerce_value("true") is True

    def test_true_mixed_case(self) -> None:
        assert _coerce_value("True") is True

    def test_true_uppercase(self) -> None:
        assert _coerce_value("TRUE") is True

    def test_false_lowercase(self) -> None:
        assert _coerce_value("false") is False

    def test_false_mixed_case(self) -> None:
        assert _coerce_value("False") is False

    def test_false_uppercase(self) -> None:
        assert _coerce_value("FALSE") is False

    def test_none_lowercase(self) -> None:
        assert _coerce_value("none") is None

    def test_none_mixed_case(self) -> None:
        assert _coerce_value("None") is None

    def test_integer(self) -> None:
        assert _coerce_value("42") == 42
        assert isinstance(_coerce_value("42"), int)

    def test_negative_integer(self) -> None:
        assert _coerce_value("-7") == -7
        assert isinstance(_coerce_value("-7"), int)

    def test_float(self) -> None:
        assert _coerce_value("3.14") == pytest.approx(3.14)
        assert isinstance(_coerce_value("3.14"), float)

    def test_negative_float(self) -> None:
        assert _coerce_value("-0.001") == pytest.approx(-0.001)

    def test_scientific_notation(self) -> None:
        assert _coerce_value("1e-4") == pytest.approx(1e-4)
        assert isinstance(_coerce_value("1e-4"), float)

    def test_string_passthrough(self) -> None:
        assert _coerce_value("hello_world") == "hello_world"
        assert isinstance(_coerce_value("hello_world"), str)

    def test_path_string_passthrough(self) -> None:
        assert _coerce_value("/some/path/file.pth") == "/some/path/file.pth"

    def test_empty_string_passthrough(self) -> None:
        assert _coerce_value("") == ""


class TestSetNested:
    """Unit tests for the _set_nested helper."""

    def test_top_level_key(self) -> None:
        data: dict[str, object] = {}
        _set_nested(data, "mode", "eval")
        assert data["mode"] == "eval"

    def test_nested_key(self) -> None:
        data: dict[str, object] = {"camera": {}}
        _set_nested(data, "camera.enable_arc_follow", True)
        assert data["camera"]["enable_arc_follow"] is True

    def test_deeply_nested_creates_intermediates(self) -> None:
        data: dict[str, object] = {}
        _set_nested(data, "a.b.c", 99)
        assert data["a"]["b"]["c"] == 99

    def test_overwrite_existing(self) -> None:
        data: dict[str, object] = {"camera": {"enable_arc_follow": False}}
        _set_nested(data, "camera.enable_arc_follow", True)
        assert data["camera"]["enable_arc_follow"] is True


class TestLoadConfigWithOverrides:
    """Test YAML loading with dotted-key overrides applied before validation."""

    @pytest.fixture()
    def base_yaml(self, tmp_path: Path) -> Path:
        content = """\
mode: train
common:
  task: navigation_task_gate
  num_envs: 256
  headless: true
camera:
  static_camera_base_y: -3.0
curriculum:
  min_level: 3
  max_level: 23
"""
        p = tmp_path / "base.yaml"
        p.write_text(content)
        return p

    def test_nested_override_camera(self, base_yaml: Path) -> None:
        cfg = load_config_with_overrides(base_yaml, {"camera.enable_arc_follow": "true"})
        assert cfg.camera.enable_arc_follow is True

    def test_override_num_envs_int_coercion(self, base_yaml: Path) -> None:
        cfg = load_config_with_overrides(base_yaml, {"common.num_envs": "512"})
        assert cfg.common.num_envs == 512

    def test_override_float_coercion(self, base_yaml: Path) -> None:
        cfg = load_config_with_overrides(base_yaml, {"camera.static_camera_base_y": "-5.5"})
        assert cfg.camera.static_camera_base_y == pytest.approx(-5.5)

    def test_override_boolean_coercion(self, base_yaml: Path) -> None:
        cfg = load_config_with_overrides(base_yaml, {"common.headless": "false"})
        assert cfg.common.headless is False

    def test_override_none_coercion(self, base_yaml: Path) -> None:
        cfg = load_config_with_overrides(base_yaml, {"common.seed": "none"})
        assert cfg.common.seed is None

    def test_override_string_passthrough(self, base_yaml: Path) -> None:
        cfg = load_config_with_overrides(base_yaml, {"common.device": "cuda:1"})
        assert cfg.common.device == "cuda:1"


class TestExtraFieldsRejected:
    """Pydantic extra='forbid' should reject unknown keys."""

    def test_extra_top_level_key_raises(self, tmp_path: Path) -> None:
        content = """\
mode: train
common:
  task: navigation_task_gate
bogus_field: 42
"""
        p = tmp_path / "bad.yaml"
        p.write_text(content)
        with pytest.raises(ValidationError, match="bogus_field"):
            load_config(str(p))

    def test_extra_nested_key_raises(self, tmp_path: Path) -> None:
        content = """\
mode: train
common:
  task: navigation_task_gate
  nonexistent_option: true
"""
        p = tmp_path / "bad2.yaml"
        p.write_text(content)
        with pytest.raises(ValidationError, match="nonexistent_option"):
            load_config(str(p))

    def test_extra_camera_key_raises(self, tmp_path: Path) -> None:
        content = """\
mode: train
common:
  task: navigation_task_gate
camera:
  totally_fake_param: 9000
"""
        p = tmp_path / "bad3.yaml"
        p.write_text(content)
        with pytest.raises(ValidationError, match="totally_fake_param"):
            load_config(str(p))


class TestMissingRequiredFields:
    """RunConfig requires common.task — missing it must fail."""

    def test_missing_common_section_raises(self) -> None:
        with pytest.raises(ValidationError, match="common"):
            RunConfig(mode="train")

    def test_missing_task_field_raises(self) -> None:
        with pytest.raises(ValidationError, match="task"):
            RunConfig(common={})

    def test_empty_yaml_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        with pytest.raises(ValidationError):
            load_config(str(p))

    def test_yaml_without_common_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "no_common.yaml"
        p.write_text("mode: train\n")
        with pytest.raises(ValidationError, match="common"):
            load_config(str(p))

    def test_config_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")
