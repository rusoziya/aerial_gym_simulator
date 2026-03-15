"""Unified entry point for training and evaluation.

Usage:
    python -m aerial_gym.run --config configs/train_gate_navigation.yaml
    python -m aerial_gym.run --config configs/train_gate_navigation.yaml --set common.num_envs=512
    python -m aerial_gym.run --config configs/eval_gate_navigation.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from aerial_gym.config.run_config import (
    RunConfig,
    _coerce_value,
    load_config,
    load_config_with_overrides,
)

AERIAL_GYM_ROOT: Path = Path(__file__).resolve().parent.parent


def _parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for the unified runner."""
    parser = argparse.ArgumentParser(
        description="Aerial Gym unified training and evaluation entry point.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values using dotted key paths (e.g. common.num_envs=512).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the command that would be executed without running it.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Validate the configuration file and exit.",
    )
    return parser.parse_args()


def _parse_overrides(raw_overrides: List[str]) -> Dict[str, object]:
    """Convert a list of 'key=value' strings into a dict of dotted-key overrides."""
    overrides: Dict[str, object] = {}
    for item in raw_overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override format (expected key=value): '{item}'")
        key, _, raw_value = item.partition("=")
        overrides[key.strip()] = _coerce_value(raw_value.strip())
    return overrides


def _print_config_summary(cfg: RunConfig) -> None:
    """Print a human-readable summary of the validated configuration."""
    print("=" * 60)
    print("Aerial Gym — Run Configuration")
    print("=" * 60)
    print(f"  Mode:        {cfg.mode.value}")
    print(f"  Framework:   {cfg.framework.value}")
    print(f"  Task:        {cfg.common.task}")
    print(f"  Num envs:    {cfg.common.num_envs}")
    print(f"  Device:      {cfg.common.device}")
    print(f"  Headless:    {cfg.common.headless}")
    print(f"  Use warp:    {cfg.common.use_warp}")
    if cfg.common.seed is not None:
        print(f"  Seed:        {cfg.common.seed}")
    if cfg.mode.value == "train":
        print(f"  Total steps: {cfg.training.total_steps:,}")
        print(f"  Batch size:  {cfg.training.batch_size}")
        print(f"  LR:          {cfg.training.learning_rate}")
        print(f"  Gamma:       {cfg.training.gamma}")
        print(f"  Checkpoint:  {cfg.training.checkpoint_dir}")
    if cfg.mode.value in ("eval", "play") and cfg.eval is not None:
        print(f"  Checkpoint:  {cfg.eval.checkpoint}")
        print(f"  Episodes:    {cfg.eval.num_episodes}")
    if cfg.wandb.enabled:
        print(f"  W&B project: {cfg.wandb.project}")
    print("=" * 60)


def _set_curriculum_env_vars(cfg: RunConfig) -> None:
    """Export curriculum-related overrides as environment variables."""
    cur = cfg.curriculum
    if cur.min_level is not None:
        os.environ["SF_MIN_CURRICULUM_LEVEL"] = str(cur.min_level)
    if cur.max_level is not None:
        os.environ["SF_MAX_CURRICULUM_LEVEL"] = str(cur.max_level)
    if cur.force_level is not None:
        os.environ["SF_FORCE_CURRICULUM_LEVEL"] = str(cur.force_level)
    if cur.disable_gate_size_randomization:
        os.environ["SF_DISABLE_GATE_SIZE_RANDOMIZATION"] = "true"
    if cur.fixed_gate_scale_percent is not None:
        os.environ["SF_FIXED_GATE_SCALE_PERCENT"] = str(cur.fixed_gate_scale_percent)
    if cur.disable_obstacle_randomization:
        os.environ["SF_DISABLE_OBSTACLE_RANDOMIZATION"] = "true"
    if cur.fixed_obstacles_behind_gate is not None:
        os.environ["SF_FIXED_OBSTACLES_BEHIND_GATE"] = str(cur.fixed_obstacles_behind_gate)


def _set_camera_env_vars(cfg: RunConfig) -> None:
    """Export camera-related overrides as environment variables."""
    cam = cfg.camera
    if cam.static_camera_base_y != -3.0:
        os.environ["SF_STATIC_CAMERA_BASE_Y"] = str(cam.static_camera_base_y)
    base_z = cam.static_camera_base_z
    if isinstance(base_z, str) and base_z == "adaptive":
        os.environ["SF_STATIC_CAMERA_BASE_Z"] = "adaptive"
    elif isinstance(base_z, (int, float)) and base_z != 1.5:
        os.environ["SF_STATIC_CAMERA_BASE_Z"] = str(float(base_z))
    if cam.disable_static_camera_orientation:
        os.environ["SF_DISABLE_STATIC_CAMERA_ORIENT_RANDOMIZATION"] = "true"
    if cam.disable_camera_noise:
        os.environ["SF_DISABLE_CAMERA_NOISE_RANDOMIZATION"] = "true"
    if cam.disable_frame_dropout:
        os.environ["SF_DISABLE_CAMERA_FRAME_DROPOUT_RANDOMIZATION"] = "true"
    if cam.enable_yaw_sweep:
        os.environ["SF_ENABLE_STATIC_CAMERA_YAW_SWEEP"] = "true"
        os.environ["SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG"] = str(cam.yaw_sweep_speed_deg)
    if cam.enable_arc_follow:
        os.environ["SF_ENABLE_STATIC_CAMERA_ARC_FOLLOW"] = "true"
        os.environ["SF_STATIC_CAMERA_ARC_RADIUS_M"] = str(cam.arc_follow_radius_m)
    if cam.enable_dynamic_following is not None:
        val = "true" if cam.enable_dynamic_following else "false"
        os.environ["enable_dynamic_camera_following"] = val


def _build_sample_factory_train_args(cfg: RunConfig) -> List[str]:
    """Build CLI argument list for Sample Factory training."""
    sf = cfg.sample_factory
    script = str(
        AERIAL_GYM_ROOT
        / "aerial_gym"
        / "rl_training"
        / "sample_factory"
        / "aerialgym_examples"
        / "train_aerialgym_custom_net_gate.py"
    )
    args: List[str] = [sys.executable, script]
    args.extend(["--env", cfg.common.task])
    args.extend(["--train_for_env_steps", str(cfg.training.total_steps)])
    args.extend(["--batch_size", str(cfg.training.batch_size)])
    args.extend(["--learning_rate", str(cfg.training.learning_rate)])
    args.extend(["--gamma", str(cfg.training.gamma)])
    args.extend(["--rollout", str(cfg.training.rollout_horizon)])
    args.extend(["--max_grad_norm", str(cfg.training.max_grad_norm)])
    args.extend(["--num_epochs", str(cfg.training.num_epochs)])
    args.extend(["--num_workers", str(sf.num_workers)])
    args.extend(["--num_envs_per_worker", str(sf.num_envs_per_worker)])
    args.extend(["--num_batches_per_epoch", str(sf.num_batches_per_epoch)])
    args.extend(["--save_best_after", str(sf.save_best_after)])
    args.extend(["--exploration_loss_coeff", str(sf.exploration_loss_coeff)])
    if sf.use_rnn:
        args.extend(["--use_rnn", "True"])
        args.extend(["--rnn_size", str(sf.rnn_size)])
        args.extend(["--rnn_type", sf.rnn_type])
        args.extend(["--rnn_num_layers", str(sf.rnn_num_layers)])
    args.extend(["--fusion", sf.fusion])
    if sf.gate_per_feature:
        args.extend(["--gate_per_feature", "True"])
    if cfg.common.seed is not None:
        args.extend(["--seed", str(cfg.common.seed)])
    if cfg.common.headless:
        args.extend(["--headless", "True"])
    if sf.experiment_name:
        args.extend(["--experiment", sf.experiment_name])
    if cfg.training.checkpoint_dir:
        args.extend(["--train_dir", cfg.training.checkpoint_dir])
    if cfg.wandb.enabled:
        args.extend(["--with_wandb", "True"])
        args.extend(["--wandb_project", cfg.wandb.project])
        if cfg.wandb.tags:
            args.extend(["--wandb_tags", " ".join(cfg.wandb.tags)])
        if cfg.wandb.entity:
            args.extend(["--wandb_entity", cfg.wandb.entity])
    return args


def _build_sample_factory_eval_args(cfg: RunConfig) -> List[str]:
    """Build CLI argument list for Sample Factory evaluation."""
    script = str(
        AERIAL_GYM_ROOT
        / "aerial_gym"
        / "rl_training"
        / "sample_factory"
        / "aerialgym_examples"
        / "enjoy_aerialgym.py"
    )
    args: List[str] = [sys.executable, script]
    args.extend(["--env", cfg.common.task])
    if cfg.eval is not None:
        args.extend(["--train_dir", str(Path(cfg.eval.checkpoint).parent)])
        if cfg.eval.save_gifs:
            args.extend(["--save_gifs", "True"])
    if cfg.common.seed is not None:
        args.extend(["--seed", str(cfg.common.seed)])
    return args


def _build_rl_games_args(cfg: RunConfig) -> List[str]:
    """Build CLI argument list for RL-Games training or evaluation."""
    script = str(AERIAL_GYM_ROOT / "aerial_gym" / "rl_training" / "rl_games" / "runner.py")
    args: List[str] = [sys.executable, script]
    args.extend(["--task", cfg.common.task])
    args.extend(["--num_envs", str(cfg.common.num_envs)])
    if cfg.common.headless:
        args.extend(["--headless", "True"])
    if cfg.common.use_warp:
        args.extend(["--use_warp", "True"])
    if cfg.common.seed is not None:
        args.extend(["--seed", str(cfg.common.seed)])
    if cfg.mode.value in ("eval", "play"):
        args.append("--play")
        if cfg.eval is not None and cfg.eval.checkpoint:
            args.extend(["--checkpoint", cfg.eval.checkpoint])
    return args


def _build_cleanrl_args(cfg: RunConfig) -> List[str]:
    """Build CLI argument list for CleanRL training."""
    script = str(
        AERIAL_GYM_ROOT / "aerial_gym" / "rl_training" / "cleanrl" / "ppo_continuous_action.py"
    )
    cr = cfg.cleanrl
    args: List[str] = [sys.executable, script]
    args.extend(["--task", cfg.common.task])
    args.extend(["--num_envs", str(cfg.common.num_envs)])
    args.extend(["--total_timesteps", str(cfg.training.total_steps)])
    args.extend(["--learning_rate", str(cfg.training.learning_rate)])
    args.extend(["--gamma", str(cfg.training.gamma)])
    args.extend(["--num_steps", str(cfg.training.rollout_horizon)])
    args.extend(["--gae_lambda", str(cr.gae_lambda)])
    args.extend(["--num_minibatches", str(cr.num_minibatches)])
    args.extend(["--update_epochs", str(cr.update_epochs)])
    args.extend(["--clip_coef", str(cr.clip_coef)])
    args.extend(["--ent_coef", str(cr.ent_coef)])
    args.extend(["--vf_coef", str(cr.vf_coef)])
    if cr.anneal_lr:
        args.extend(["--anneal_lr", "True"])
    if cfg.common.seed is not None:
        args.extend(["--seed", str(cfg.common.seed)])
    if cfg.common.headless:
        args.extend(["--headless", "True"])
    return args


def _build_command(cfg: RunConfig) -> List[str]:
    """Dispatch to the correct framework command builder."""
    framework = cfg.framework.value
    mode = cfg.mode.value

    if framework == "sample_factory":
        if mode == "train":
            return _build_sample_factory_train_args(cfg)
        return _build_sample_factory_eval_args(cfg)

    if framework == "rl_games":
        return _build_rl_games_args(cfg)

    if framework == "cleanrl":
        if mode in ("eval", "play"):
            print(
                "Warning: CleanRL does not have a dedicated eval script. "
                "Building training command instead."
            )
        return _build_cleanrl_args(cfg)

    raise ValueError(f"Unsupported framework: {framework}")


def main() -> int:
    """Entry point for `python -m aerial_gym.run`."""
    cli_args = _parse_cli_args()

    overrides = _parse_overrides(cli_args.set)
    if overrides:
        cfg = load_config_with_overrides(cli_args.config, overrides)
    else:
        cfg = load_config(cli_args.config)

    if cli_args.validate_only:
        _print_config_summary(cfg)
        print("Configuration is valid.")
        return 0

    _print_config_summary(cfg)
    _set_curriculum_env_vars(cfg)
    _set_camera_env_vars(cfg)

    cmd = _build_command(cfg)

    if cli_args.dry_run:
        print("\n[dry-run] Would execute:")
        print("  " + " ".join(cmd))
        return 0

    print(f"\nExecuting: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
