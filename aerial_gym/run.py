"""Unified entry point for Sample Factory training and evaluation.

Usage:
    python -m aerial_gym.run --config configs/train_gate_sf.yaml
    python -m aerial_gym.run --config configs/train_gate_sf.yaml --set common.num_envs=512
    python -m aerial_gym.run --config configs/eval_gate_drone_only.yaml
    python -m aerial_gym.run --config configs/eval_gate_all_modalities.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from aerial_gym.config.run_config import (
    RunConfig,
    _coerce_value,
    load_config,
    load_config_with_overrides,
)
from aerial_gym.config.run_config_enums import Mode

AERIAL_GYM_ROOT: Path = Path(__file__).resolve().parent.parent


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aerial Gym — Sample Factory training and evaluation runner.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values (e.g. common.num_envs=512).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing.")
    parser.add_argument("--validate-only", action="store_true", help="Validate config and exit.")
    parser.add_argument(
        "--log",
        action="store_true",
        help="Save all output (stdout+stderr) to a timestamped log file.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for log files (default: ./logs).",
    )
    return parser.parse_args()


def _parse_overrides(raw_overrides: List[str]) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    for item in raw_overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override format (expected key=value): '{item}'")
        key, _, raw_value = item.partition("=")
        overrides[key.strip()] = _coerce_value(raw_value.strip())
    return overrides


def _print_config_summary(cfg: RunConfig) -> None:
    print("=" * 60)
    print("Aerial Gym — Run Configuration")
    print("=" * 60)
    print(f"  Mode:        {cfg.mode.value}")
    print(f"  Task:        {cfg.common.task.value}")
    print(f"  Num envs:    {cfg.common.num_envs}")
    print(f"  Device:      {cfg.common.device}")
    print(f"  Headless:    {cfg.common.headless}")
    print(f"  Use warp:    {cfg.common.use_warp}")
    if cfg.common.seed is not None:
        print(f"  Seed:        {cfg.common.seed}")
    if cfg.mode == Mode.train:
        print(f"  Total steps: {cfg.training.total_steps:,}")
        print(f"  Batch size:  {cfg.training.batch_size}")
        print(f"  LR:          {cfg.training.learning_rate}")
        print(f"  Gamma:       {cfg.training.gamma}")
        print(f"  Fusion:      {cfg.sample_factory.fusion}")
        print(f"  Checkpoint:  {cfg.training.checkpoint_dir}")
    if cfg.mode in (Mode.eval, Mode.play) and cfg.eval is not None:
        print(f"  Checkpoint:  {cfg.eval.checkpoint}")
        print(f"  Episodes:    {cfg.eval.num_episodes}")
    if cfg.mode == Mode.inference_suite and cfg.inference_suite is not None:
        s = cfg.inference_suite
        print(f"  Seeds:       {s.seeds}")
        print(f"  Levels:      {s.curriculum_levels}")
        total = len(s.seeds) * len(s.curriculum_levels)
        print(f"  Total runs:  {total}")
    if cfg.ablation.obs_ranges:
        print(f"  Ablation:    {cfg.ablation.obs_ranges}")
    if cfg.wandb.enabled:
        print(f"  W&B project: {cfg.wandb.project}")
    print("=" * 60)


def _setenv(key: str, value: bool) -> None:
    os.environ[key] = "true" if value else "false"


def _set_camera_env_vars(cfg: RunConfig) -> None:
    """Set all camera-related environment variables."""
    cam = cfg.camera

    _setenv("SF_DISABLE_STATIC_CAMERA_ORIENT_RANDOMIZATION", cam.disable_static_camera_orientation)
    _setenv("SF_DISABLE_CAMERA_NOISE_RANDOMIZATION", cam.disable_camera_noise)
    _setenv("SF_DISABLE_CAMERA_FRAME_DROPOUT_RANDOMIZATION", cam.disable_frame_dropout)
    _setenv("SF_DISABLE_STATE_NOISE_RANDOMIZATION", cam.disable_state_noise)

    if cam.disable_drone_camera_noise is not None:
        _setenv("SF_DISABLE_DRONE_CAMERA_NOISE_RANDOMIZATION", cam.disable_drone_camera_noise)
    if cam.disable_static_camera_noise is not None:
        _setenv("SF_DISABLE_STATIC_CAMERA_NOISE_RANDOMIZATION", cam.disable_static_camera_noise)
    if cam.disable_drone_camera_frame_dropout is not None:
        _setenv("SF_DISABLE_DRONE_CAMERA_FRAME_DROPOUT", cam.disable_drone_camera_frame_dropout)
    if cam.disable_static_camera_frame_dropout is not None:
        _setenv("SF_DISABLE_STATIC_CAMERA_FRAME_DROPOUT", cam.disable_static_camera_frame_dropout)

    _setenv("SF_ENABLE_STATIC_CAMERA_YAW_SWEEP", cam.enable_yaw_sweep)
    _setenv("SF_STATIC_CAMERA_LOCKED_FOLLOW", cam.enable_locked_follow)
    _setenv("SF_ENABLE_STATIC_CAMERA_ARC_FOLLOW", cam.enable_arc_follow)
    os.environ["SF_STATIC_CAMERA_YAW_SWEEP_SPEED_DEG"] = str(cam.yaw_sweep_speed_deg)
    os.environ["SF_STATIC_CAMERA_ARC_RADIUS_M"] = str(cam.arc_follow_radius_m)

    os.environ["SF_STATIC_CAMERA_BASE_Y"] = str(cam.static_camera_base_y)
    base_z = cam.static_camera_base_z
    os.environ["SF_STATIC_CAMERA_BASE_Z"] = (
        "adaptive" if isinstance(base_z, str) and base_z.lower() == "adaptive" else str(base_z)
    )
    if cam.dynamic_camera_follow_y_offset_m is not None:
        os.environ["SF_DYNAMIC_CAMERA_FOLLOW_OFFSET_Y"] = str(cam.dynamic_camera_follow_y_offset_m)
    _setenv("SF_DISABLE_DYNAMIC_FOLLOW_GATE_BLENDING", cam.disable_dynamic_follow_gate_blending)

    if cam.enable_dynamic_following is not None:
        _setenv("enable_dynamic_camera_following", cam.enable_dynamic_following)
        _setenv("disable_dynamic_camera_following", not cam.enable_dynamic_following)


def _set_curriculum_env_vars(cfg: RunConfig) -> None:
    """Set curriculum, spawn, gate, and obstacle environment variables."""
    cur = cfg.curriculum

    _setenv("SF_DISABLE_SPAWN_POSITION_RANDOMIZATION", cur.disable_spawn_position)
    _setenv("SF_DISABLE_SPAWN_ORIENTATION_RANDOMIZATION", cur.disable_spawn_orientation)

    _setenv("SF_DISABLE_CURRICULUM_MULTIPLIER", cur.disable_curriculum_multiplier)
    if cur.force_level is not None:
        os.environ["SF_FORCE_CURRICULUM_LEVEL"] = str(cur.force_level)
    if cur.min_level is not None:
        os.environ["SF_MIN_CURRICULUM_LEVEL"] = str(cur.min_level)
    if cur.max_level is not None:
        os.environ["SF_MAX_CURRICULUM_LEVEL"] = str(cur.max_level)

    _setenv("SF_DISABLE_GATE_SIZE_RANDOMIZATION", cur.disable_gate_size_randomization)
    if cur.fixed_gate_scale_percent is not None:
        os.environ["SF_FIXED_GATE_SCALE_PERCENT"] = str(cur.fixed_gate_scale_percent)
    _setenv("SF_DISABLE_OBSTACLE_RANDOMIZATION", cur.disable_obstacle_randomization)
    if cur.fixed_obstacles_behind_gate is not None:
        os.environ["SF_FIXED_OBSTACLES_BEHIND_GATE"] = str(cur.fixed_obstacles_behind_gate)

    if cfg.ablation.obs_ranges:
        os.environ["ABLATE_OBS_RANGES"] = cfg.ablation.obs_ranges
    _setenv("ABLATE_ZERO_RNN", cfg.ablation.zero_rnn)


def _set_training_env_vars(cfg: RunConfig) -> None:
    """Set fusion, gradient monitoring, eval stretch, and agent environment variables."""
    sf = cfg.sample_factory

    _setenv("SF_ENABLE_INFLUENCE_TRACKER", cfg.gradient_monitoring.enable_influence_tracker)
    _setenv("SF_ENABLE_GRAD_ATTR", cfg.gradient_monitoring.enable_grad_attribution)

    os.environ["SF_FUSION_MODE"] = sf.fusion.value
    os.environ["SF_GATE_PER_FEATURE"] = "1" if sf.gate_per_feature else "0"

    if cfg.eval is not None and cfg.eval.eval_stretch_enabled:
        os.environ["EVAL_STRETCH_ENABLED"] = "1"
        os.environ["EVAL_STRETCH_END_LEVEL"] = str(cfg.eval.eval_stretch_end_level)

    os.environ["SF_ENV_AGENTS"] = str(cfg.common.num_envs)
    _setenv("SF_HEADLESS", cfg.common.headless)
    os.environ["SF_TRAIN_DIR"] = sf.train_dir
    os.environ["SF_EXPERIMENT_NAME"] = sf.experiment_name

    os.environ["SF_LOG_LEVEL"] = cfg.logging.log_level.value

    if cfg.logging.wandb_dir_override is not None:
        wandb_dir = cfg.logging.wandb_dir_override
    else:
        experiment_dir = Path(sf.train_dir)
        if sf.experiment_name:
            experiment_dir = experiment_dir / sf.experiment_name
        wandb_dir = str(experiment_dir)
    os.environ["WANDB_DIR"] = wandb_dir


def _set_env_vars_from_config(cfg: RunConfig) -> None:
    """Set ALL environment variables matching cfg_env_bridge.py behavior."""
    _set_camera_env_vars(cfg)
    _set_curriculum_env_vars(cfg)
    _set_training_env_vars(cfg)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["AERIAL_GYM_BACKEND"] = cfg.common.backend.value


def _get_sf_train_script(cfg: RunConfig) -> str:
    """Select the correct SF training script based on task name."""
    task = cfg.common.task
    base = AERIAL_GYM_ROOT / "aerial_gym" / "rl_training" / "sample_factory" / "aerialgym_examples"
    if "gate" in task:
        return str(base / "train_aerialgym_custom_net_gate.py")
    if "quad_with_obstacles" in cfg.sample_factory.env_name:
        return str(base / "train_aerialgym_custom_net.py")
    return str(base / "train_aerialgym.py")


def _build_train_args(cfg: RunConfig) -> List[str]:
    """Build the full CLI argument list for SF training."""
    sf = cfg.sample_factory
    t = cfg.training
    c = cfg.common

    args: List[str] = [sys.executable, _get_sf_train_script(cfg)]
    args.extend(
        [
            f"--env={sf.env_name.value}",
            f"--experiment={sf.experiment_name}",
            f"--train_dir={sf.train_dir}",
            f"--num_workers={sf.num_workers}",
            f"--num_envs_per_worker={sf.num_envs_per_worker}",
            f"--env_agents={c.num_envs}",
            f"--obs_key={sf.obs_key.value}",
            f"--batch_size={t.batch_size}",
            f"--num_batches_to_accumulate={sf.num_batches_to_accumulate}",
            f"--num_batches_per_epoch={sf.num_batches_per_epoch}",
            f"--num_epochs={t.num_epochs}",
            f"--rollout={t.rollout_horizon}",
            f"--learning_rate={t.learning_rate}",
            f"--use_rnn={str(sf.use_rnn).lower()}",
            f"--rnn_size={sf.rnn_size}",
            f"--rnn_num_layers={sf.rnn_num_layers}",
        ]
    )

    args.append("--encoder_mlp_layers")
    args.extend(str(x) for x in sf.encoder_mlp_layers)

    args.extend(
        [
            f"--gamma={t.gamma}",
            f"--reward_scale={sf.reward_scale}",
            f"--max_grad_norm={t.max_grad_norm}",
            f"--normalize_input={str(sf.normalize_input).lower()}",
            "--use_env_info_cache=false",
            f"--with_wandb={str(cfg.wandb.enabled).lower()}",
            f"--wandb_project={cfg.wandb.project}",
            f"--wandb_user={cfg.wandb.user}",
            f"--wandb_group={cfg.wandb.group}",
        ]
    )
    if cfg.wandb.tags:
        args.append("--wandb_tags")
        args.extend(cfg.wandb.tags)

    args.extend(
        [
            f"--save_every_sec={sf.save_every_sec}",
            f"--save_best_every_sec={sf.save_best_every_sec}",
            f"--train_for_env_steps={t.total_steps}",
            f"--train_for_seconds={sf.train_for_seconds}",
            f"--async_rl={str(sf.async_rl).lower()}",
            f"--serial_mode={str(sf.serial_mode).lower()}",
            f"--policy_workers_per_policy={sf.policy_workers_per_policy}",
            f"--headless={str(c.headless).lower()}",
            f"--fusion={sf.fusion.value}",
            f"--gate_per_feature={'1' if sf.gate_per_feature else '0'}",
        ]
    )

    if c.seed is not None:
        args.extend(["--seed", str(c.seed)])
    if cfg.curriculum.force_level is not None:
        args.extend(["--force_curriculum_level", str(cfg.curriculum.force_level)])
    if sf.load_checkpoint_kind.value:
        args.extend([f"--load_checkpoint_kind={sf.load_checkpoint_kind.value}"])
    args.extend([f"--restart_behavior={sf.restart_behavior.value}"])

    return args


def _build_eval_args(cfg: RunConfig) -> List[str]:
    """Build CLI argument list for SF evaluation."""
    base = AERIAL_GYM_ROOT / "aerial_gym" / "rl_training" / "sample_factory" / "aerialgym_examples"
    script = str(base / "enjoy_aerialgym.py")
    args: List[str] = [sys.executable, script]

    sf = cfg.sample_factory
    c = cfg.common
    args.extend([f"--env={sf.env_name.value}", f"--env_agents={c.num_envs}"])
    args.extend([f"--headless={str(c.headless).lower()}"])

    if cfg.eval is not None:
        if cfg.eval.checkpoint:
            checkpoint_path = Path(cfg.eval.checkpoint)
            args.extend([f"--train_dir={checkpoint_path.parent}"])
        args.extend([f"--max_num_episodes={cfg.eval.num_episodes}"])
        args.extend([f"--eval_deterministic={str(cfg.eval.eval_deterministic).lower()}"])
        if cfg.eval.save_gifs:
            args.append("--save_gifs=true")

    if sf.experiment_name:
        args.extend([f"--experiment={sf.experiment_name}"])
    if c.seed is not None:
        args.extend(["--seed", str(c.seed)])

    return args


def _run_inference_suite(cfg: RunConfig, dry_run: bool) -> int:
    """Run batch inference across all seed x level combinations."""
    suite = cfg.inference_suite
    if suite is None:
        raise ValueError("inference_suite config is required for inference_suite mode")

    total_runs = len(suite.seeds) * len(suite.curriculum_levels)
    run_idx = 0
    failed_runs: List[str] = []

    for seed in suite.seeds:
        for level in suite.curriculum_levels:
            run_idx += 1
            run_name = f"L{level}_SEED{seed}"
            print(f"\n{'=' * 60}")
            print(f"[{run_idx}/{total_runs}] {run_name}")
            print(f"{'=' * 60}")

            os.environ["SF_FORCE_CURRICULUM_LEVEL"] = str(level)
            if cfg.wandb.enabled:
                os.environ["WANDB_RUN_NAME"] = run_name

            # Build eval args with this seed
            eval_cfg = cfg.model_copy(
                update={"common": cfg.common.model_copy(update={"seed": seed})}
            )
            args = _build_eval_args(eval_cfg)
            args.extend([f"--force_curriculum_level={level}"])

            if dry_run:
                print("  [dry-run] " + " ".join(args))
                continue

            print(f"  Executing: {' '.join(args)}")
            result = subprocess.run(args)
            if result.returncode != 0:
                failed_runs.append(run_name)
                print(f"  [WARN] {run_name} exited with code {result.returncode}")

    if failed_runs:
        print(f"\n{len(failed_runs)}/{total_runs} runs failed: {', '.join(failed_runs)}")
        return 1
    print(f"\nAll {total_runs} inference runs completed successfully.")
    return 0


def _build_command(cfg: RunConfig) -> List[str]:
    if cfg.mode == Mode.train:
        return _build_train_args(cfg)
    return _build_eval_args(cfg)


def _make_log_path(log_dir: str, cfg: RunConfig) -> Path:
    """Generate a timestamped log file path inside the training directory."""
    # Use train_dir from config so logs, checkpoints, and wandb are co-located
    base_dir = Path(cfg.sample_factory.train_dir)
    if cfg.sample_factory.experiment_name:
        base_dir = base_dir / cfg.sample_factory.experiment_name
    log_path = base_dir / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task = cfg.common.task.value
    mode = cfg.mode.value
    return log_path / f"{mode}_{task}_{ts}.log"


def _get_git_hash() -> str:
    """Get short git hash of current HEAD."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _run_with_logging(cmd: List[str], log_file: Path | None, config_yaml: str = "") -> int:
    """Run a subprocess, tee-ing stdout+stderr to a log file if provided."""
    if log_file is None:
        result = subprocess.run(cmd)
        return result.returncode

    print(f"Logging to: {log_file}\n")
    with open(log_file, "w") as fh:
        fh.write(f"Command: {' '.join(cmd)}\n")
        fh.write(f"Started: {datetime.now().isoformat()}\n")
        fh.write(f"Git hash: {_get_git_hash()}\n")
        fh.write(f"Experiment: {os.environ.get('SF_EXPERIMENT_NAME', '')}\n")
        fh.write(f"Wandb dir: {os.environ.get('WANDB_DIR', '')}\n")
        if config_yaml:
            fh.write(f"\n--- Config ---\n{config_yaml}--- End Config ---\n")
        fh.write("=" * 80 + "\n\n")
        fh.flush()

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        for line in iter(proc.stdout.readline, b""):
            decoded = line.decode("utf-8", errors="replace")
            sys.stdout.write(decoded)
            sys.stdout.flush()
            fh.write(decoded)
            fh.flush()
        proc.wait()

        fh.write(f"\n{'=' * 80}\n")
        fh.write(f"Finished: {datetime.now().isoformat()}\n")
        fh.write(f"Exit code: {proc.returncode}\n")

    return proc.returncode


def main() -> int:
    cli_args = _parse_cli_args()

    overrides = _parse_overrides(cli_args.set)
    if overrides:
        cfg = load_config_with_overrides(cli_args.config, overrides)
    else:
        cfg = load_config(cli_args.config)

    # Auto-generate experiment name if empty
    if not cfg.sample_factory.experiment_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        task = cfg.common.task.value
        mode = cfg.mode.value
        auto_name = f"{task}_{mode}_{ts}"
        cfg = cfg.model_copy(
            update={
                "sample_factory": cfg.sample_factory.model_copy(
                    update={"experiment_name": auto_name}
                )
            }
        )
        print(f"Auto-generated experiment name: {auto_name}")

    if cli_args.validate_only:
        _print_config_summary(cfg)
        print("Configuration is valid.")
        return 0

    _print_config_summary(cfg)
    _set_env_vars_from_config(cfg)

    log_file = _make_log_path(cli_args.log_dir, cfg) if cli_args.log else None

    if cfg.mode == Mode.inference_suite:
        return _run_inference_suite(cfg, cli_args.dry_run)

    cmd = _build_command(cfg)

    if cli_args.dry_run:
        print("\n[dry-run] Would execute:")
        print("  " + " ".join(cmd))
        return 0

    print(f"\nExecuting: {' '.join(cmd)}\n")
    # Embed the config YAML in the log for reproducibility
    config_yaml = ""
    if log_file is not None:
        config_path = Path(cli_args.config)
        if config_path.exists():
            config_yaml = config_path.read_text()
    return _run_with_logging(cmd, log_file, config_yaml)


if __name__ == "__main__":
    sys.exit(main())
