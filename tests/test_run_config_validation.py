"""Tests for RunConfig Pydantic validation — no Isaac Gym required.

Uses importlib to load run_config directly, bypassing aerial_gym.__init__
which imports isaacgym (unavailable on non-GPU / wrong Python version).
"""

from __future__ import annotations

import importlib.util
import sys

import pytest
from pydantic import ValidationError

# Load run_config modules directly to avoid aerial_gym.__init__ importing isaacgym
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

CurriculumOverrides = _config_mod.CurriculumOverrides
LoggingConfig = _config_mod.LoggingConfig
RunConfig = _config_mod.RunConfig
load_config = _config_mod.load_config
LogLevel = _enums_mod.LogLevel


class TestCurriculumValidation:
    def test_min_greater_than_max_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_level.*must be <= max_level"):
            CurriculumOverrides(min_level=30, max_level=5)

    def test_force_with_min_raises(self) -> None:
        with pytest.raises(ValidationError, match="force_level cannot coexist"):
            CurriculumOverrides(force_level=10, min_level=3)

    def test_force_with_max_raises(self) -> None:
        with pytest.raises(ValidationError, match="force_level cannot coexist"):
            CurriculumOverrides(force_level=10, max_level=23)

    def test_force_with_both_raises(self) -> None:
        with pytest.raises(ValidationError, match="force_level cannot coexist"):
            CurriculumOverrides(force_level=10, min_level=3, max_level=23)

    def test_valid_min_max_passes(self) -> None:
        cfg = CurriculumOverrides(min_level=3, max_level=23)
        assert cfg.min_level == 3
        assert cfg.max_level == 23

    def test_equal_min_max_passes(self) -> None:
        cfg = CurriculumOverrides(min_level=10, max_level=10)
        assert cfg.min_level == cfg.max_level

    def test_force_alone_passes(self) -> None:
        cfg = CurriculumOverrides(force_level=13)
        assert cfg.force_level == 13
        assert cfg.min_level is None
        assert cfg.max_level is None

    def test_all_none_passes(self) -> None:
        cfg = CurriculumOverrides()
        assert cfg.force_level is None
        assert cfg.min_level is None
        assert cfg.max_level is None


class TestLoggingConfig:
    def test_defaults(self) -> None:
        cfg = LoggingConfig()
        assert cfg.log_level == LogLevel.info
        assert cfg.log_to_file is True
        assert cfg.wandb_dir_override is None

    def test_debug_level(self) -> None:
        cfg = LoggingConfig(log_level="DEBUG")
        assert cfg.log_level == LogLevel.debug

    def test_warning_level(self) -> None:
        cfg = LoggingConfig(log_level="WARNING")
        assert cfg.log_level == LogLevel.warning

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValidationError):
            LoggingConfig(log_level="TRACE")

    def test_wandb_dir_override(self) -> None:
        cfg = LoggingConfig(wandb_dir_override="/custom/wandb")
        assert cfg.wandb_dir_override == "/custom/wandb"


class TestLogLevelEnum:
    def test_values(self) -> None:
        assert LogLevel.debug.value == "DEBUG"
        assert LogLevel.info.value == "INFO"
        assert LogLevel.warning.value == "WARNING"

    def test_from_string(self) -> None:
        assert LogLevel("DEBUG") == LogLevel.debug
        assert LogLevel("INFO") == LogLevel.info


class TestRunConfigLoggingSection:
    def test_logging_defaults_in_runconfig(self) -> None:
        cfg = RunConfig(common={"task": "navigation_task_gate"})
        assert cfg.logging.log_level == LogLevel.info
        assert cfg.logging.log_to_file is True

    def test_logging_override_in_yaml(self, tmp_path) -> None:
        yaml_content = """
mode: train
common:
  task: navigation_task_gate
logging:
  log_level: DEBUG
  log_to_file: false
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(str(config_file))
        assert cfg.logging.log_level == LogLevel.debug
        assert cfg.logging.log_to_file is False


CameraOverrides = _config_mod.CameraOverrides
GradientMonitoringConfig = _config_mod.GradientMonitoringConfig
Mode = _enums_mod.Mode


class TestCameraModeExclusivity:
    """Camera modes are mutually exclusive — at most one can be active."""

    def test_two_modes_raises(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            CameraOverrides(enable_arc_follow=True, enable_yaw_sweep=True)

    def test_three_modes_raises(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            CameraOverrides(
                enable_arc_follow=True,
                enable_yaw_sweep=True,
                enable_locked_follow=True,
            )

    def test_dynamic_and_arc_raises(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            CameraOverrides(enable_dynamic_following=True, enable_arc_follow=True)

    def test_dynamic_and_locked_raises(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            CameraOverrides(enable_dynamic_following=True, enable_locked_follow=True)

    def test_single_mode_arc_passes(self) -> None:
        cfg = CameraOverrides(enable_arc_follow=True)
        assert cfg.enable_arc_follow is True

    def test_single_mode_sweep_passes(self) -> None:
        cfg = CameraOverrides(enable_yaw_sweep=True)
        assert cfg.enable_yaw_sweep is True

    def test_all_false_passes(self) -> None:
        cfg = CameraOverrides()
        assert cfg.enable_arc_follow is False
        assert cfg.enable_yaw_sweep is False
        assert cfg.enable_locked_follow is False
        assert cfg.enable_dynamic_following is None


class TestDynamicFollowYOffsetRequiresEnable:
    """dynamic_camera_follow_y_offset_m without enable_dynamic_following must fail."""

    def test_y_offset_without_dynamic_raises(self) -> None:
        with pytest.raises(ValidationError, match="dynamic_camera_follow_y_offset_m"):
            RunConfig(
                common={"task": "navigation_task_gate"},
                camera={"dynamic_camera_follow_y_offset_m": -1.0},
            )

    def test_y_offset_with_dynamic_passes(self) -> None:
        cfg = RunConfig(
            common={"task": "navigation_task_gate"},
            camera={
                "enable_dynamic_following": True,
                "dynamic_camera_follow_y_offset_m": -1.0,
            },
        )
        assert cfg.camera.dynamic_camera_follow_y_offset_m == -1.0

    def test_gate_blending_without_dynamic_raises(self) -> None:
        with pytest.raises(ValidationError, match="disable_dynamic_follow_gate_blending"):
            RunConfig(
                common={"task": "navigation_task_gate"},
                camera={"disable_dynamic_follow_gate_blending": True},
            )


class TestGradientMonitoringEvalMode:
    """Gradient monitoring in eval/play/inference_suite mode must raise."""

    def test_influence_tracker_in_eval_raises(self) -> None:
        with pytest.raises(ValidationError, match="Gradient monitoring"):
            RunConfig(
                mode="eval",
                common={"task": "navigation_task_gate"},
                eval={"checkpoint": "dummy.pth"},
                gradient_monitoring={"enable_influence_tracker": True},
            )

    def test_grad_attribution_in_play_raises(self) -> None:
        with pytest.raises(ValidationError, match="Gradient monitoring"):
            RunConfig(
                mode="play",
                common={"task": "navigation_task_gate"},
                eval={"checkpoint": "dummy.pth"},
                gradient_monitoring={"enable_grad_attribution": True},
            )

    def test_grad_monitoring_in_inference_suite_raises(self) -> None:
        with pytest.raises(ValidationError, match="Gradient monitoring"):
            RunConfig(
                mode="inference_suite",
                common={"task": "navigation_task_gate"},
                inference_suite={},
                eval={"checkpoint": "dummy.pth"},
                gradient_monitoring={"enable_influence_tracker": True},
            )

    def test_grad_monitoring_in_train_passes(self) -> None:
        cfg = RunConfig(
            common={"task": "navigation_task_gate"},
            gradient_monitoring={
                "enable_influence_tracker": True,
                "enable_grad_attribution": True,
            },
        )
        assert cfg.gradient_monitoring.enable_influence_tracker is True
        assert cfg.gradient_monitoring.enable_grad_attribution is True
