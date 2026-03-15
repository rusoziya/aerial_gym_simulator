from __future__ import annotations

import math
from typing import Callable

from aerial_gym.rl_training.sample_factory.aerialgym_examples.influence_metric_utils import (
    OBS_GRAD_PREFIXES,
    compute_obs_grad_shares,
    compute_obs_pct_by_window,
    drop_nonfinite_episode_stats,
    is_obs_grad_key,
    metrics_to_float,
    sanitize_metrics,
)

from sample_factory.utils.typing import Config


def build_enhanced_wandb_log(
    cfg: Config,
    original_wandb_log: Callable[..., None],
    get_influence_tracker: Callable[[], object | None],
    get_grad_tracker: Callable[[], object | None],
    last_obsgrad_influence: dict[str, float],
    last_obsgrad_grad: dict[str, float],
) -> Callable[..., None]:
    """Build the enhanced wandb.log wrapper that injects influence metrics."""

    def enhanced_wandb_log(metrics: dict[str, object] | None, **kwargs: object) -> None:
        if metrics is None:
            metrics = {}
        else:
            metrics = dict(metrics)

        _inject_frames(cfg, metrics, kwargs)

        influence_tracker = get_influence_tracker()
        if influence_tracker:
            _merge_influence_metrics(influence_tracker, cfg, metrics, last_obsgrad_influence)
        else:
            pass

        grad_tracker = get_grad_tracker()
        if grad_tracker and grad_tracker.should_log():
            _merge_grad_metrics(grad_tracker, metrics, last_obsgrad_grad)

        sanitize_metrics(metrics)
        drop_nonfinite_episode_stats(metrics)

        for k in list(metrics.keys()):
            if isinstance(k, str) and k.startswith("curriculum/"):
                del metrics[k]

        _define_wandb_metrics(metrics)

        if original_wandb_log:
            original_wandb_log(metrics, **kwargs)

    return enhanced_wandb_log


def _inject_frames(cfg: Config, metrics: dict[str, object], kwargs: dict[str, object]) -> None:
    """Attach frames / env_steps from cfg if available."""
    frames: int | None = None
    train_step = getattr(cfg, "train_step", None)
    if isinstance(train_step, (int, float)):
        frames = int(train_step)
    else:
        env_steps = getattr(cfg, "env_steps", None)
        if isinstance(env_steps, (int, float)):
            frames = int(env_steps)
    if frames is not None:
        metrics.setdefault("frames", frames)
        kwargs.setdefault("step", frames)


def _merge_influence_metrics(
    tracker: object,
    cfg: Config,
    metrics: dict[str, object],
    cache: dict[str, float],
) -> None:
    """Merge influence tracker metrics into the W&B payload."""
    influence_metrics = metrics_to_float(tracker.get_logging_metrics())  # type: ignore[attr-defined]
    metrics.update(influence_metrics)

    had_obs = any(is_obs_grad_key(k) for k in influence_metrics)
    if had_obs:
        cache.clear()
        cache.update(influence_metrics)

    source = influence_metrics if influence_metrics else cache
    _add_episode_extra_obs_grad(source, metrics)

    if tracker.should_log():  # type: ignore[attr-defined]
        tracker.step()  # type: ignore[attr-defined]
        step_count = getattr(tracker, "step_count", None)
        if step_count is not None and step_count % cfg.gradient_print_interval == 0:
            tracker.print_analysis_summary()  # type: ignore[attr-defined]


def _add_episode_extra_obs_grad(source: dict[str, float], metrics: dict[str, object]) -> None:
    """Mirror obs_grad metrics under episode_extra_stats and compute derived shares."""
    episode_extra: dict[str, float] = {}
    for name, val in source.items():
        if is_obs_grad_key(name):
            prefix_removed = name.split("/", 1)[1] if "/" in name else name
            episode_extra["episode_extra_stats/obs_grad/" + prefix_removed] = float(val)
    if episode_extra:
        metrics.update(episode_extra)

    shares = compute_obs_grad_shares(source)
    for k, v in shares.items():
        metrics["episode_extra_stats/obs_grad/" + k] = v

    try:
        pct = compute_obs_pct_by_window(source)
        for k, v in pct.items():
            metrics["episode_extra_stats/obs_grad/" + k] = v
    except (ValueError, TypeError):
        pass


def _merge_grad_metrics(
    tracker: object,
    metrics: dict[str, object],
    cache: dict[str, float],
) -> None:
    """Merge gradient attribution tracker metrics into the W&B payload."""
    grad_metrics = metrics_to_float(tracker.get_logging_metrics())  # type: ignore[attr-defined]
    metrics.update(grad_metrics)

    mirrored: dict[str, float] = {}
    for name, val in grad_metrics.items():
        if is_obs_grad_key(name):
            prefix_removed = name.split("/", 1)[1] if "/" in name else name
            mirrored["episode_extra_stats/obs_grad/" + prefix_removed] = val
    if mirrored:
        metrics.update(mirrored)
        cache.clear()
        cache.update(grad_metrics)


def _define_wandb_metrics(metrics: dict[str, object]) -> None:
    """Call wandb.define_metric for known custom groups."""
    import wandb

    if not hasattr(wandb, "define_metric"):
        return
    wandb.define_metric("frames")
    for name in list(metrics.keys()):
        if isinstance(name, str) and name.startswith(
            (
                "obs_grad/",
                "influence/",
                "gpu/",
                "reward_breakdown/",
                "episode_extra_stats/obs_grad/",
                "episode_extra_stats/curriculum/",
            )
        ):
            wandb.define_metric(name, step_metric="frames")
    for key in (
        "episode_extra_stats/path_efficiency",
        "episode_extra_stats/time_to_gate_steps",
        "episode_extra_stats/min_gate_distance",
        "episode_extra_stats/center_offset_success",
        "episode_extra_stats/height_offset_success",
    ):
        wandb.define_metric(key, step_metric="frames")
