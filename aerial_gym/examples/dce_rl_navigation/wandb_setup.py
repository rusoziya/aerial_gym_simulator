"""W&B initialization and metric registration for gate navigation inference."""

from __future__ import annotations

import os


def init_wandb(
    cfg: object,
    args: object,
) -> tuple[object | None, bool]:
    """Initialise Weights & Biases for inference logging.

    Returns (wandb_run, use_wandb).
    """
    wandb_run = None
    try:
        use_wandb = os.environ.get("WANDB_DISABLED", "true").lower() == "false"
    except RuntimeError:
        use_wandb = False
    if not use_wandb:
        return None, False
    try:
        import wandb

        project, entity = _resolve_project_entity(cfg, args)
        run_name = _resolve_run_name(cfg, args)
        mode = os.environ.get("WANDB_MODE", "online")
        dir_path = args.wandb_dir or os.environ.get("WANDB_DIR", "")
        if dir_path:
            wandb_run = wandb.init(
                project=project, entity=entity, name=run_name, mode=mode, dir=dir_path
            )
        else:
            wandb_run = wandb.init(project=project, entity=entity, name=run_name, mode=mode)
        _register_metrics(wandb)
    except RuntimeError:
        wandb_run = None
    return wandb_run, use_wandb


def _resolve_project_entity(cfg: object, args: object) -> tuple[str, str | None]:
    project = (
        args.wandb_project
        or os.environ.get("WANDB_PROJECT", "")
        or cfg.wandb_project
        or "gate_eval_runs"
    )
    entity = (args.wandb_entity or os.environ.get("WANDB_ENTITY", "") or cfg.wandb_user) or None
    return project, entity


def _resolve_run_name(cfg: object, args: object) -> str:
    cli_run_name = args.run_name or ""
    env_run_name = os.environ.get("WANDB_RUN_NAME", "")
    try:
        fallback_name = f"eval_{cfg.experiment}"
    except AttributeError:
        fallback_name = f"eval_{cfg.algo}"
    if len(cli_run_name) > 0:
        return cli_run_name
    if len(env_run_name) > 0:
        return env_run_name
    return fallback_name


def _register_metrics(wandb: object) -> None:
    """Define W&B step metrics and metric-to-step mappings."""
    try:
        wandb.define_metric("episodes")
        wandb.define_metric("episodes_batch")
        wandb.define_metric("global_step")
        wandb.define_metric("frames")
        for name in _EPISODE_STEP_METRICS:
            wandb.define_metric(name, step_metric="episodes")
        for name in _VISIBILITY_STEP_METRICS:
            wandb.define_metric(name, step_metric="episodes")
        wandb.define_metric("episodes/success_rate_batch", step_metric="episodes_batch")
    except RuntimeError:
        pass


_EPISODE_STEP_METRICS: tuple[str, ...] = (
    "curriculum/level",
    "curriculum/progress",
    "episode_extra_stats/path_efficiency",
    "episode_extra_stats/time_to_gate_steps",
    "episode_extra_stats/min_gate_distance",
    "episode_extra_stats/center_offset_success",
    "episode_extra_stats/height_offset_success",
    "episode_extra_stats/gate_pass_rate",
    "gate_pass_rate_running_mean",
    "episode_extra_stats/target_success_rate",
    "episode_extra_stats/successes",
    "episode_extra_stats/crashes",
    "episode_extra_stats/timeouts",
    "path_efficiency_running_mean",
    "time_to_gate_running_mean",
    "min_gate_distance_running_mean",
    "center_offset_running_mean",
    "height_offset_running_mean",
    "target_success_running_mean",
    "episode_extra_stats/episodes_total",
    "episode_extra_stats/episodes_crossed",
    "curriculum/total_successes",
    "curriculum/total_crashes",
    "curriculum/total_timeouts",
    "curriculum/total_resets",
    "curriculum/success_rate",
    "curriculum/crash_rate",
    "curriculum/timeout_rate",
    "episodes/success_binary",
    "episode_extra_stats/episode_return",
    "episode_return_mean",
    "episode_return_std",
    "episode_length",
    "episode_length_mean",
    "success_only/episode_return_mean",
    "success_only/episode_length_mean",
    "success_rate_running",
    "early_crash_rate",
    "action_abs_mean/x",
    "action_abs_mean/y",
    "action_abs_mean/z",
    "action_abs_mean/yaw",
    "action_diff_mean/x",
    "action_diff_mean/y",
    "action_diff_mean/z",
    "action_diff_mean/yaw",
    "action_saturation_rate",
    "episode_action_diff_mean/x",
    "episode_action_diff_mean/y",
    "episode_action_diff_mean/z",
    "episode_action_diff_mean/yaw",
    "vae/drone_norm_mean",
    "vae/static_norm_mean",
    "vae/static_to_drone_norm_ratio",
    "throughput/fps_env",
    "throughput/episodes_per_min",
    "spatial/center_offset_p50",
    "spatial/center_offset_p90",
    "spatial/height_offset_p50",
    "spatial/height_offset_p90",
    "spatial/min_gate_distance_p50",
    "spatial/min_gate_distance_p90",
)

_VISIBILITY_STEP_METRICS: tuple[str, ...] = (
    "visibility/abs_running_mean",
    "visibility/frustum_running_mean",
    "visibility/eff_running_mean",
    "fov/score_running_mean",
    "fov/visible_rate_running_mean",
    "fov/horiz_angle_rad_running_mean",
    "fov/vert_angle_rad_running_mean",
)
