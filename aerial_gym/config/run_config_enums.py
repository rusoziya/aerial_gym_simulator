"""Enum types for the aerial_gym run configuration schema."""

from __future__ import annotations

from enum import Enum


class Mode(str, Enum):
    train = "train"
    eval = "eval"
    play = "play"
    inference_suite = "inference_suite"


class FusionMode(str, Enum):
    gated = "gated"
    concat = "concat"


class RNNType(str, Enum):
    gru = "gru"
    lstm = "lstm"


class Nonlinearity(str, Enum):
    elu = "elu"
    relu = "relu"
    tanh = "tanh"


class LRSchedule(str, Enum):
    kl_adaptive_epoch = "kl_adaptive_epoch"
    kl_adaptive_minibatch = "kl_adaptive_minibatch"
    linear_decay = "linear_decay"
    constant = "constant"


class RestartBehavior(str, Enum):
    resume = "resume"
    restart = "restart"


class CheckpointKind(str, Enum):
    best = "best"
    latest = "latest"
    none = ""


class PolicyInit(str, Enum):
    torch_default = "torch_default"
    xavier_uniform = "xavier_uniform"
    orthogonal = "orthogonal"


class ObsKey(str, Enum):
    """Observation key in the obs dict passed to the policy."""

    observations = "observations"
    obs = "obs"


class TaskName(str, Enum):
    """Known task names from the task registry."""

    position_setpoint_task = "position_setpoint_task"
    navigation_task = "navigation_task"
    navigation_task_gate = "navigation_task_gate"
    dce_navigation_task_gate = "dce_navigation_task_gate"


class SFEnvName(str, Enum):
    """Known Sample Factory environment names."""

    position_setpoint_task = "position_setpoint_task"
    navigation_task = "navigation_task"
    quad_with_obstacles = "quad_with_obstacles"
    quad_with_obstacles_gate = "quad_with_obstacles_gate"
    dce_navigation_task_gate = "dce_navigation_task_gate"


class CameraBaseZ(str, Enum):
    """Special string value for adaptive camera Z positioning."""

    adaptive = "adaptive"


class LogLevel(str, Enum):
    """Python logging level."""

    debug = "DEBUG"
    info = "INFO"
    warning = "WARNING"


class PhysicsBackendType(str, Enum):
    """Physics simulation backend."""

    isaacgym = "isaacgym"
    isaaclab = "isaaclab"
