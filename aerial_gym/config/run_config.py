"""Pydantic-validated configuration schema for the aerial_gym Sample Factory pipeline.

Load configs from YAML with optional CLI overrides. All training, evaluation, and
inference-suite modes are supported through a single RunConfig schema.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from aerial_gym.config.run_config_enums import (
    CheckpointKind,
    FusionMode,
    LogLevel,
    LRSchedule,
    Mode,
    Nonlinearity,
    ObsKey,
    PolicyInit,
    RestartBehavior,
    RNNType,
    SFEnvName,
    TaskName,
)


class _StrictBase(BaseModel):
    """Base model that forbids extra fields to catch typos in YAML configs."""

    model_config = {"extra": "forbid"}


class CommonConfig(_StrictBase):
    """Shared environment parameters."""

    task: TaskName = Field(..., description="Task name from task_registry")
    num_envs: int = Field(256, ge=1, description="Number of parallel environments")
    seed: Optional[int] = Field(None, description="Random seed (None = unseeded)")
    device: str = Field("cuda:0", description="Compute device")
    headless: bool = Field(True, description="Run without viewer")
    use_warp: bool = Field(True, description="Use Warp rendering pipeline")


class TrainingConfig(_StrictBase):
    """Training hyperparameters (framework-agnostic subset)."""

    total_steps: int = Field(100_000_000, ge=1, description="Total environment steps")
    batch_size: int = Field(2048, ge=1)
    learning_rate: float = Field(3e-4, gt=0)
    gamma: float = Field(0.99, ge=0, le=1, description="Discount factor")
    rollout_horizon: int = Field(32, ge=1)
    max_grad_norm: float = Field(1.0, gt=0)
    num_epochs: int = Field(4, ge=1)
    checkpoint_interval: int = Field(1_000_000, ge=1)
    checkpoint_dir: str = Field("./train_dir")
    resume_from: Optional[str] = Field(None, description="Path to checkpoint to resume from")


class SampleFactoryConfig(_StrictBase):
    """Sample Factory framework parameters."""

    # Environment naming
    env_name: SFEnvName = Field(
        SFEnvName.quad_with_obstacles_gate, description="SF --env parameter"
    )
    experiment_name: str = Field("", description="Experiment folder name (auto-generated if empty)")
    train_dir: str = Field("./train_dir")
    obs_key: ObsKey = Field(ObsKey.observations, description="Observation key in obs dict")

    # Workers and parallelism
    num_workers: int = Field(1, ge=1)
    num_envs_per_worker: int = Field(1, ge=1)
    async_rl: bool = Field(False)
    serial_mode: bool = Field(True)
    policy_workers_per_policy: int = Field(1, ge=1)
    batched_sampling: bool = Field(True)
    worker_num_splits: int = Field(1, ge=1)
    actor_worker_gpus: List[int] = Field(default_factory=lambda: [0])

    # RNN
    use_rnn: bool = Field(True)
    rnn_size: int = Field(64, ge=1)
    rnn_type: RNNType = Field(RNNType.gru)
    rnn_num_layers: int = Field(1, ge=1)

    # Encoder and fusion
    encoder_mlp_layers: List[int] = Field(default_factory=lambda: [512, 256, 64])
    fusion: FusionMode = Field(FusionMode.gated)
    gate_per_feature: bool = Field(True)

    # PPO hyperparameters
    num_batches_per_epoch: int = Field(4, ge=1)
    num_batches_to_accumulate: int = Field(1, ge=1)
    exploration_loss_coeff: float = Field(0.001, ge=0)
    adaptive_stddev: bool = Field(True)
    policy_initialization: PolicyInit = Field(PolicyInit.torch_default)
    env_gpu_actions: bool = Field(True)
    env_gpu_observations: bool = Field(True)
    reward_scale: float = Field(0.1, gt=0)
    ppo_clip_ratio: float = Field(0.2, gt=0)
    value_loss_coeff: float = Field(2.0, gt=0)
    nonlinearity: Nonlinearity = Field(Nonlinearity.elu)
    kl_loss_coeff: float = Field(0.1, ge=0)
    gae_lambda: float = Field(0.95, ge=0, le=1)
    with_vtrace: bool = Field(False)
    value_bootstrap: bool = Field(True)

    # Learning rate schedule
    lr_schedule: LRSchedule = Field(LRSchedule.kl_adaptive_epoch)
    lr_schedule_kl_threshold: float = Field(0.016, gt=0)

    # Normalization
    normalize_input: bool = Field(True)
    normalize_returns: bool = Field(True)
    shuffle_minibatches: bool = Field(False)

    # Checkpointing
    save_best_after: int = Field(100_000, ge=0)
    save_every_sec: int = Field(1800, ge=0)
    save_best_every_sec: int = Field(500, ge=0)
    restart_behavior: RestartBehavior = Field(RestartBehavior.resume)
    load_checkpoint_kind: CheckpointKind = Field(CheckpointKind.none)

    # Time limit
    train_for_seconds: int = Field(0, ge=0, description="0 = no time limit")


class EvalConfig(_StrictBase):
    """Evaluation / inference parameters."""

    checkpoint: str = Field("", description="Path to trained checkpoint .pth")
    num_episodes: int = Field(100, ge=1)
    save_gifs: bool = Field(False)
    gif_dir: str = Field("./gif_episodes")
    rnn_warmup_steps: int = Field(50, ge=0)
    dump_obs_parity: bool = Field(False)
    eval_deterministic: bool = Field(True)
    eval_stretch_enabled: bool = Field(False)
    eval_stretch_end_level: int = Field(33, ge=0)


class WandbConfig(_StrictBase):
    """Weights & Biases logging configuration."""

    enabled: bool = Field(False)
    project: str = Field("aerial-gym")
    entity: str = Field("")
    user: str = Field("")
    run_name: str = Field("")
    group: str = Field("")
    tags: List[str] = Field(default_factory=list)


class CurriculumOverrides(_StrictBase):
    """Curriculum and spawn/obstacle randomization overrides."""

    # Level control
    force_level: Optional[int] = Field(None, ge=0, description="Force specific curriculum level")
    min_level: Optional[int] = Field(None, ge=0)
    max_level: Optional[int] = Field(None, ge=0)
    disable_curriculum_multiplier: bool = Field(False)

    # Spawn randomization
    disable_spawn_position: bool = Field(False)
    disable_spawn_orientation: bool = Field(False)

    # Obstacle randomization
    disable_obstacle_randomization: bool = Field(False)
    fixed_obstacles_behind_gate: Optional[int] = Field(None, ge=0)

    # Gate randomization
    disable_gate_size_randomization: bool = Field(False)
    fixed_gate_scale_percent: Optional[int] = Field(
        None, ge=40, le=100, description="Even integer in [40, 100]"
    )

    @model_validator(mode="after")
    def _validate_level_constraints(self) -> CurriculumOverrides:
        if self.min_level is not None and self.max_level is not None:
            if self.min_level > self.max_level:
                raise ValueError(
                    f"min_level ({self.min_level}) must be <= max_level ({self.max_level})"
                )
        if self.force_level is not None and (
            self.min_level is not None or self.max_level is not None
        ):
            raise ValueError(
                "force_level cannot coexist with min_level/max_level — "
                "force_level pins the curriculum, min/max constrain progression"
            )
        return self


class CameraOverrides(_StrictBase):
    """Camera mode and noise configuration.

    Camera modes are mutually exclusive — at most one of enable_dynamic_following,
    enable_yaw_sweep, enable_locked_follow, enable_arc_follow can be True.
    Default (all False) = static randomized orientation.
    """

    # Global noise/dropout toggles
    disable_static_camera_orientation: bool = Field(False)
    disable_camera_noise: bool = Field(False)
    disable_frame_dropout: bool = Field(False)
    disable_state_noise: bool = Field(False)

    # Per-camera noise/dropout (None = inherit from global)
    disable_drone_camera_noise: Optional[bool] = Field(None)
    disable_static_camera_noise: Optional[bool] = Field(None)
    disable_drone_camera_frame_dropout: Optional[bool] = Field(None)
    disable_static_camera_frame_dropout: Optional[bool] = Field(None)

    # Camera modes (mutually exclusive)
    enable_dynamic_following: Optional[bool] = Field(None, description="Dynamic follow drone+gate")
    enable_yaw_sweep: bool = Field(False, description="Oscillating yaw sweep")
    enable_locked_follow: bool = Field(False, description="Lock camera to drone position")
    enable_arc_follow: bool = Field(False, description="Arc orbit around gate")

    # Camera position parameters
    static_camera_base_y: float = Field(-3.0)
    static_camera_base_z: Union[float, str] = Field(1.5, description="Float or 'adaptive'")
    yaw_sweep_speed_deg: float = Field(10.0, gt=0)
    arc_follow_radius_m: float = Field(2.0, gt=0)
    dynamic_camera_follow_y_offset_m: Optional[float] = Field(None)
    disable_dynamic_follow_gate_blending: bool = Field(False)

    @field_validator("static_camera_base_z")
    @classmethod
    def _validate_base_z(cls, v: Union[float, str]) -> Union[float, str]:
        if isinstance(v, str) and v != "adaptive":
            raise ValueError("static_camera_base_z must be a number or 'adaptive'")
        return v

    @model_validator(mode="after")
    def _enforce_camera_mode_exclusivity(self) -> CameraOverrides:
        """Ensure at most one camera mode is active."""
        active_modes = []
        if self.enable_dynamic_following is True:
            active_modes.append("enable_dynamic_following")
        if self.enable_yaw_sweep:
            active_modes.append("enable_yaw_sweep")
        if self.enable_locked_follow:
            active_modes.append("enable_locked_follow")
        if self.enable_arc_follow:
            active_modes.append("enable_arc_follow")
        if len(active_modes) > 1:
            raise ValueError(
                f"Camera modes are mutually exclusive — only one can be active, "
                f"but got: {', '.join(active_modes)}"
            )
        return self


class AblationConfig(_StrictBase):
    """Observation ablation for controlled experiments."""

    obs_ranges: str = Field("", description="e.g. '0:22=zero,86:150=zero'")
    zero_rnn: bool = Field(False, description="Zero out RNN hidden state")


class GradientMonitoringConfig(_StrictBase):
    """Gradient monitoring for observation influence tracking."""

    enable_influence_tracker: bool = Field(False)
    enable_grad_attribution: bool = Field(False)
    log_interval: int = Field(100, ge=1)
    print_interval: int = Field(100, ge=1)


class InferenceSuiteConfig(_StrictBase):
    """Batch inference across multiple seeds and curriculum levels."""

    seeds: List[int] = Field(default_factory=lambda: [123, 231, 321, 456, 789])
    curriculum_levels: List[int] = Field(default_factory=lambda: [3, 13, 23, 33])
    max_episodes_per_run: int = Field(512, ge=1)
    eval_deterministic: bool = Field(True)
    checkpoint_path: str = Field("", description="Path to .pth checkpoint file")
    policy_dir: str = Field("", description="Parent dir containing experiment folders")
    experiment_name: str = Field("")


class LoggingConfig(_StrictBase):
    """Logging configuration."""

    log_level: LogLevel = Field(LogLevel.info, description="Python log level for all loggers")
    log_to_file: bool = Field(True, description="Write logger output to file in experiment dir")
    wandb_dir_override: Optional[str] = Field(
        None, description="Override WANDB_DIR. None = co-locate with train_dir/experiment/"
    )


class RunConfig(_StrictBase):
    """Top-level run configuration for aerial_gym Sample Factory pipeline."""

    mode: Mode = Field(Mode.train)
    common: CommonConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: Optional[EvalConfig] = Field(None)
    sample_factory: SampleFactoryConfig = Field(default_factory=SampleFactoryConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    curriculum: CurriculumOverrides = Field(default_factory=CurriculumOverrides)
    camera: CameraOverrides = Field(default_factory=CameraOverrides)
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    gradient_monitoring: GradientMonitoringConfig = Field(default_factory=GradientMonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    inference_suite: Optional[InferenceSuiteConfig] = Field(None)

    @model_validator(mode="after")
    def _check_mode_requirements(self) -> RunConfig:
        if self.mode in (Mode.eval, Mode.play) and self.eval is None:
            raise ValueError("'eval' config section is required when mode is 'eval' or 'play'")
        if self.mode == Mode.inference_suite and self.inference_suite is None:
            raise ValueError(
                "'inference_suite' config section is required when mode is 'inference_suite'"
            )
        return self

    @model_validator(mode="after")
    def _check_cross_field_consistency(self) -> RunConfig:
        """Validate that fields make sense together."""
        # Arc follow requires a radius
        cam = self.camera
        if cam.enable_arc_follow and cam.arc_follow_radius_m <= 0:
            raise ValueError("arc_follow_radius_m must be > 0 when enable_arc_follow is True")

        # Dynamic follow Y offset only makes sense when dynamic following is enabled
        if cam.dynamic_camera_follow_y_offset_m is not None and not cam.enable_dynamic_following:
            raise ValueError(
                "dynamic_camera_follow_y_offset_m requires enable_dynamic_following=True"
            )

        # Gate blending disable only relevant with dynamic following
        if cam.disable_dynamic_follow_gate_blending and not cam.enable_dynamic_following:
            raise ValueError(
                "disable_dynamic_follow_gate_blending requires enable_dynamic_following=True"
            )

        # Gradient monitoring makes no sense in eval mode
        gm = self.gradient_monitoring
        if self.mode in (Mode.eval, Mode.play, Mode.inference_suite):
            if gm.enable_influence_tracker or gm.enable_grad_attribution:
                raise ValueError("Gradient monitoring is only supported in train mode")

        # Fixed gate scale only meaningful when gate randomization is disabled
        cur = self.curriculum
        if cur.fixed_gate_scale_percent is not None and not cur.disable_gate_size_randomization:
            raise ValueError(
                "fixed_gate_scale_percent requires disable_gate_size_randomization=True"
            )

        return self


def _set_nested(data: Dict[str, object], dotted_key: str, value: object) -> None:
    """Set a value in a nested dict using a dotted key path."""
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
    """Load a RunConfig from a YAML file, validated by Pydantic."""
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
    """Load a RunConfig from YAML, then apply dotted-key overrides before validation."""
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
