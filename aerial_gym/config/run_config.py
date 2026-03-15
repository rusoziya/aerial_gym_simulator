"""Pydantic-validated configuration schema for the aerial_gym training and evaluation pipeline.

Supports all three RL frameworks (rl_games, sample_factory, cleanrl) with a unified interface.
Load configs from YAML with optional CLI overrides.
"""

from __future__ import annotations

import copy
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

try:
    from pydantic import BaseModel as _BaseModel
    from pydantic import field_validator, model_validator

    PYDANTIC_V2 = True
except ImportError:
    try:
        from pydantic import BaseModel as _BaseModel
        from pydantic import validator

        PYDANTIC_V2 = False
    except ImportError as exc:
        raise ImportError(
            "pydantic is required for run_config. Install with: pip install pydantic"
        ) from exc


class _StrictBase(_BaseModel):
    """Base model that forbids extra fields to catch typos."""

    if PYDANTIC_V2:

        class Config:
            extra = "forbid"

    else:

        class Config:
            extra = "forbid"


class Framework(str, Enum):
    rl_games = "rl_games"
    sample_factory = "sample_factory"
    cleanrl = "cleanrl"


class Mode(str, Enum):
    train = "train"
    eval = "eval"
    play = "play"


class CommonConfig(_StrictBase):
    """Shared parameters across all frameworks."""

    task: str
    num_envs: int = 256
    seed: Optional[int] = None
    device: str = "cuda:0"
    headless: bool = True
    use_warp: bool = True

    if PYDANTIC_V2:

        @field_validator("num_envs")
        @classmethod
        def _num_envs_positive(cls, v: int) -> int:
            if v < 1:
                raise ValueError("num_envs must be >= 1")
            return v

    else:

        @validator("num_envs")
        @classmethod
        def _num_envs_positive(cls, v: int) -> int:
            if v < 1:
                raise ValueError("num_envs must be >= 1")
            return v


class TrainingConfig(_StrictBase):
    """Training hyperparameters."""

    total_steps: int = 100_000_000
    batch_size: int = 2048
    learning_rate: float = 3e-4
    gamma: float = 0.99
    rollout_horizon: int = 32
    max_grad_norm: float = 1.0
    num_epochs: int = 4
    checkpoint_interval: int = 1_000_000
    checkpoint_dir: str = "./train_dir"
    resume_from: Optional[str] = None

    if PYDANTIC_V2:

        @field_validator(
            "total_steps", "batch_size", "rollout_horizon", "num_epochs", "checkpoint_interval"
        )
        @classmethod
        def _positive_int(cls, v: int) -> int:
            if v < 1:
                raise ValueError("value must be >= 1")
            return v

        @field_validator("learning_rate", "max_grad_norm")
        @classmethod
        def _positive_float(cls, v: float) -> float:
            if v <= 0:
                raise ValueError("value must be > 0")
            return v

        @field_validator("gamma")
        @classmethod
        def _gamma_range(cls, v: float) -> float:
            if not 0 <= v <= 1:
                raise ValueError("gamma must be in [0, 1]")
            return v

    else:

        @validator(
            "total_steps", "batch_size", "rollout_horizon", "num_epochs", "checkpoint_interval"
        )
        @classmethod
        def _positive_int(cls, v: int) -> int:
            if v < 1:
                raise ValueError("value must be >= 1")
            return v

        @validator("learning_rate", "max_grad_norm")
        @classmethod
        def _positive_float(cls, v: float) -> float:
            if v <= 0:
                raise ValueError("value must be > 0")
            return v

        @validator("gamma")
        @classmethod
        def _gamma_range(cls, v: float) -> float:
            if not 0 <= v <= 1:
                raise ValueError("gamma must be in [0, 1]")
            return v


class SampleFactoryConfig(_StrictBase):
    """Sample Factory specific parameters."""

    num_workers: int = 1
    num_envs_per_worker: int = 1
    use_rnn: bool = True
    rnn_size: int = 64
    rnn_type: str = "gru"
    rnn_num_layers: int = 1
    encoder_mlp_layers: List[int] = [256, 128, 64]
    fusion: str = "gated"
    gate_per_feature: bool = True
    num_batches_per_epoch: int = 4
    exploration_loss_coeff: float = 0.0
    save_best_after: int = 100_000
    experiment_name: str = ""


class RlGamesConfig(_StrictBase):
    """RL-Games specific parameters."""

    config_file: str = "ppo_aerial_quad.yaml"
    experiment_name: str = ""


class CleanRLConfig(_StrictBase):
    """CleanRL specific parameters."""

    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    anneal_lr: bool = True
    experiment_name: str = ""

    if PYDANTIC_V2:

        @field_validator("gae_lambda")
        @classmethod
        def _lambda_range(cls, v: float) -> float:
            if not 0 <= v <= 1:
                raise ValueError("gae_lambda must be in [0, 1]")
            return v

        @field_validator("clip_coef", "vf_coef")
        @classmethod
        def _positive(cls, v: float) -> float:
            if v <= 0:
                raise ValueError("value must be > 0")
            return v

        @field_validator("ent_coef")
        @classmethod
        def _non_negative(cls, v: float) -> float:
            if v < 0:
                raise ValueError("ent_coef must be >= 0")
            return v

    else:

        @validator("gae_lambda")
        @classmethod
        def _lambda_range(cls, v: float) -> float:
            if not 0 <= v <= 1:
                raise ValueError("gae_lambda must be in [0, 1]")
            return v

        @validator("clip_coef", "vf_coef")
        @classmethod
        def _positive(cls, v: float) -> float:
            if v <= 0:
                raise ValueError("value must be > 0")
            return v

        @validator("ent_coef")
        @classmethod
        def _non_negative(cls, v: float) -> float:
            if v < 0:
                raise ValueError("ent_coef must be >= 0")
            return v


class EvalConfig(_StrictBase):
    """Evaluation / inference parameters."""

    checkpoint: str
    num_episodes: int = 100
    save_gifs: bool = False
    gif_dir: str = "./gif_episodes"
    rnn_warmup_steps: int = 50
    dump_obs_parity: bool = False

    if PYDANTIC_V2:

        @field_validator("num_episodes")
        @classmethod
        def _episodes_positive(cls, v: int) -> int:
            if v < 1:
                raise ValueError("num_episodes must be >= 1")
            return v

        @field_validator("rnn_warmup_steps")
        @classmethod
        def _warmup_non_negative(cls, v: int) -> int:
            if v < 0:
                raise ValueError("rnn_warmup_steps must be >= 0")
            return v

    else:

        @validator("num_episodes")
        @classmethod
        def _episodes_positive(cls, v: int) -> int:
            if v < 1:
                raise ValueError("num_episodes must be >= 1")
            return v

        @validator("rnn_warmup_steps")
        @classmethod
        def _warmup_non_negative(cls, v: int) -> int:
            if v < 0:
                raise ValueError("rnn_warmup_steps must be >= 0")
            return v


class WandbConfig(_StrictBase):
    """Weights & Biases logging configuration."""

    enabled: bool = False
    project: str = "aerial-gym"
    entity: str = ""
    run_name: str = ""
    tags: List[str] = []


class CurriculumOverrides(_StrictBase):
    """Optional curriculum overrides."""

    force_level: Optional[int] = None
    min_level: Optional[int] = None
    max_level: Optional[int] = None
    disable_obstacle_randomization: bool = False
    fixed_obstacles_behind_gate: Optional[int] = None
    disable_gate_size_randomization: bool = False
    fixed_gate_scale_percent: Optional[int] = None

    if PYDANTIC_V2:

        @field_validator("force_level", "min_level", "max_level", "fixed_obstacles_behind_gate")
        @classmethod
        def _non_negative_optional(cls, v: Optional[int]) -> Optional[int]:
            if v is not None and v < 0:
                raise ValueError("value must be >= 0")
            return v

        @field_validator("fixed_gate_scale_percent")
        @classmethod
        def _gate_scale_range(cls, v: Optional[int]) -> Optional[int]:
            if v is not None and not 40 <= v <= 100:
                raise ValueError("fixed_gate_scale_percent must be in [40, 100]")
            return v

    else:

        @validator("force_level", "min_level", "max_level", "fixed_obstacles_behind_gate")
        @classmethod
        def _non_negative_optional(cls, v: Optional[int]) -> Optional[int]:
            if v is not None and v < 0:
                raise ValueError("value must be >= 0")
            return v

        @validator("fixed_gate_scale_percent")
        @classmethod
        def _gate_scale_range(cls, v: Optional[int]) -> Optional[int]:
            if v is not None and not 40 <= v <= 100:
                raise ValueError("fixed_gate_scale_percent must be in [40, 100]")
            return v


class CameraOverrides(_StrictBase):
    """Optional camera configuration overrides."""

    disable_static_camera_orientation: bool = False
    disable_camera_noise: bool = False
    disable_frame_dropout: bool = False
    enable_dynamic_following: Optional[bool] = None
    enable_yaw_sweep: bool = False
    yaw_sweep_speed_deg: float = 10.0
    static_camera_base_y: float = -3.0
    static_camera_base_z: Union[float, str] = 1.5
    enable_arc_follow: bool = False
    arc_follow_radius_m: float = 2.0

    if PYDANTIC_V2:

        @field_validator("static_camera_base_z")
        @classmethod
        def _validate_base_z(cls, v: Union[float, str]) -> Union[float, str]:
            if isinstance(v, str) and v != "adaptive":
                raise ValueError("static_camera_base_z string value must be 'adaptive'")
            return v

    else:

        @validator("static_camera_base_z")
        @classmethod
        def _validate_base_z(cls, v: Union[float, str]) -> Union[float, str]:
            if isinstance(v, str) and v != "adaptive":
                raise ValueError("static_camera_base_z string value must be 'adaptive'")
            return v


class RunConfig(_StrictBase):
    """Top-level run configuration for aerial_gym training and evaluation."""

    mode: Mode = Mode.train
    framework: Framework = Framework.sample_factory
    common: CommonConfig
    training: TrainingConfig = TrainingConfig()
    eval: Optional[EvalConfig] = None
    sample_factory: SampleFactoryConfig = SampleFactoryConfig()
    rl_games: RlGamesConfig = RlGamesConfig()
    cleanrl: CleanRLConfig = CleanRLConfig()
    wandb: WandbConfig = WandbConfig()
    curriculum: CurriculumOverrides = CurriculumOverrides()
    camera: CameraOverrides = CameraOverrides()

    if PYDANTIC_V2:

        @model_validator(mode="after")
        def _eval_required_for_eval_mode(self) -> RunConfig:
            if self.mode in (Mode.eval, Mode.play) and self.eval is None:
                raise ValueError("eval config is required when mode is 'eval' or 'play'")
            return self

    else:

        @validator("eval", always=True)
        @classmethod
        def _eval_required_for_eval_mode(
            cls,
            v: Optional[EvalConfig],
            values: Dict[str, object],
        ) -> Optional[EvalConfig]:
            mode = values.get("mode")
            if mode in (Mode.eval, Mode.play) and v is None:
                raise ValueError("eval config is required when mode is 'eval' or 'play'")
            return v


# Pydantic v1 requires update_forward_refs() when using `from __future__ import annotations`
if not PYDANTIC_V2:
    _locals = {
        "CommonConfig": CommonConfig,
        "TrainingConfig": TrainingConfig,
        "SampleFactoryConfig": SampleFactoryConfig,
        "RlGamesConfig": RlGamesConfig,
        "CleanRLConfig": CleanRLConfig,
        "EvalConfig": EvalConfig,
        "WandbConfig": WandbConfig,
        "CurriculumOverrides": CurriculumOverrides,
        "CameraOverrides": CameraOverrides,
        "RunConfig": RunConfig,
        "Mode": Mode,
        "Framework": Framework,
    }
    for _cls in _locals.values():
        if hasattr(_cls, "update_forward_refs"):
            _cls.update_forward_refs(**_locals)


def _set_nested(data: Dict[str, object], dotted_key: str, value: object) -> None:
    """Set a value in a nested dict using a dotted key path (e.g. 'training.learning_rate')."""
    keys = dotted_key.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]  # type: ignore[assignment]
    current[keys[-1]] = value  # type: ignore[index]


def _coerce_value(raw: str) -> object:
    """Best-effort coerce a CLI string value to a Python scalar."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.lower() == "none":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def load_config(path: Union[str, Path]) -> RunConfig:
    """Load a RunConfig from a YAML file, validated by Pydantic.

    Args:
        path: Filesystem path to the YAML configuration file.

    Returns:
        A fully validated RunConfig instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML is malformed.
        pydantic.ValidationError: If the config fails validation.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as fh:
        raw_data: Dict[str, object] = yaml.safe_load(fh) or {}

    return RunConfig(**raw_data)  # type: ignore[arg-type]


def load_config_with_overrides(
    path: Union[str, Path],
    overrides: Dict[str, object],
) -> RunConfig:
    """Load a RunConfig from YAML, then apply dict overrides before validation.

    Overrides use dotted key paths (e.g. ``{"training.learning_rate": 1e-3}``).
    String values from CLI are auto-coerced to bool/int/float where possible.

    Args:
        path: Filesystem path to the YAML configuration file.
        overrides: Mapping of dotted-key paths to override values.

    Returns:
        A fully validated RunConfig instance with overrides applied.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as fh:
        raw_data: Dict[str, object] = yaml.safe_load(fh) or {}

    data = copy.deepcopy(raw_data)
    for key, value in overrides.items():
        coerced = _coerce_value(value) if isinstance(value, str) else value
        _set_nested(data, key, coerced)

    return RunConfig(**data)  # type: ignore[arg-type]
