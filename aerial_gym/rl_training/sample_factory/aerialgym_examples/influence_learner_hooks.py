from __future__ import annotations

import math
import os
from typing import Callable

import torch

from sample_factory.utils.typing import Config

from aerial_gym.rl_training.sample_factory.aerialgym_examples.influence_metric_utils import (
    CURRICULUM_KEYS,
    compute_windowed_shares,
    get_last_any,
    get_last_stat,
    is_obs_grad_key,
    metrics_to_float,
)


def create_enhanced_learner_init(
    original_init: Callable[..., object],
    cfg: Config,
    tracker_config: dict[str, int],
    grad_config: dict[str, int],
    create_influence_tracker: Callable[..., object],
    create_gradient_tracker: Callable[..., object],
    tracker_state: dict[str, object],
) -> Callable[..., object]:
    """Return a patched Learner.init that attaches influence/gradient trackers."""

    def enhanced_learner_init(self: object) -> object:
        result = original_init(self)

        if self.actor_critic is not None:  # type: ignore[attr-defined]
            _attach_influence_tracker(
                self, cfg, tracker_config, create_influence_tracker, tracker_state
            )
            _attach_grad_tracker(self, cfg, grad_config, create_gradient_tracker, tracker_state)
            _attach_forward_backward_hooks(self, tracker_state)
        else:
            print("Cannot create influence/gradient trackers - model not available")

        self._influence_tracker = tracker_state.get("influence")  # type: ignore[attr-defined]
        self._grad_tracker = tracker_state.get("grad")  # type: ignore[attr-defined]

        _emit_initial_curriculum_keys(self)
        return result

    return enhanced_learner_init


def _attach_influence_tracker(
    learner: object,
    cfg: Config,
    tracker_config: dict[str, int],
    create_influence_tracker: Callable[..., object],
    tracker_state: dict[str, object],
) -> None:
    env_val = os.getenv("SF_ENABLE_INFLUENCE_TRACKER")
    enable = bool(cfg.enable_gradient_monitoring)
    if env_val is not None:
        enable = str(env_val).lower() == "true"
    if not enable:
        tracker_state["influence"] = None
        return
    try:
        tracker = create_influence_tracker(learner.actor_critic, tracker_config)  # type: ignore[attr-defined]
        if tracker and tracker.enabled:
            print("Influence tracker successfully attached to model")
            print(f"  Logging every {cfg.gradient_log_interval} steps")
        tracker_state["influence"] = tracker
    except (ValueError, TypeError) as e:
        print(f"Error creating influence tracker: {e}")
        tracker_state["influence"] = None


def _attach_grad_tracker(
    learner: object,
    cfg: Config,
    grad_config: dict[str, int],
    create_gradient_tracker: Callable[..., object],
    tracker_state: dict[str, object],
) -> None:
    env_val = os.getenv("SF_ENABLE_GRAD_ATTR")
    enable = bool(cfg.enable_grad_attribution)
    if env_val is not None:
        enable = str(env_val).lower() == "true"
    if not enable:
        tracker_state["grad"] = None
        return
    try:
        tracker = create_gradient_tracker(learner.actor_critic, grad_config)  # type: ignore[attr-defined]
        if tracker and tracker.enabled:
            print("Gradient attribution tracker successfully attached")
        tracker_state["grad"] = tracker
    except (ValueError, TypeError) as e:
        print(f"Error creating gradient attribution tracker: {e}")
        tracker_state["grad"] = None


def _attach_forward_backward_hooks(learner: object, tracker_state: dict[str, object]) -> None:
    """Attach forward/backward hooks to intercept the 150D obs tensor for grad attribution."""
    grad_tracker = tracker_state.get("grad")
    if not grad_tracker or not grad_tracker.enabled:
        return

    try:

        def _ac_forward_hook(
            mod: torch.nn.Module, inp: tuple[object, ...]
        ) -> tuple[object, ...] | dict[str, object] | torch.Tensor | None:
            return _process_forward_input(mod, inp, learner)

        encoder = getattr(learner.actor_critic, "encoder", None)  # type: ignore[attr-defined]
        if encoder is None or "ScriptModule" in str(type(encoder)):
            target = learner.actor_critic  # type: ignore[attr-defined]
        else:
            target = encoder
        learner._grad_attr_forward_handle = target.register_forward_pre_hook(_ac_forward_hook)  # type: ignore[attr-defined]

        def _ac_backward_hook(
            mod: torch.nn.Module,
            grad_in: tuple[torch.Tensor | None, ...],
            grad_out: tuple[torch.Tensor | None, ...],
        ) -> None:
            x = getattr(mod, "_obs_proxy", None)
            gt = getattr(learner, "_grad_tracker", None)
            if x is not None and x.grad is not None and gt:
                gt.consume_grad(x.grad)

        learner._grad_attr_backward_handle = target.register_full_backward_hook(  # type: ignore[attr-defined]
            _ac_backward_hook
        )
    except (KeyError, TypeError) as e:
        print(f"Failed to attach actor_critic grad mirror hooks: {e}")


def _process_forward_input(
    mod: torch.nn.Module,
    inp: tuple[object, ...] | object,
    learner: object,
) -> tuple[object, ...] | dict[str, object] | torch.Tensor | None:
    """Process forward hook input, intercepting the 150D obs for gradient tracking."""
    try:
        arg = inp[0] if isinstance(inp, tuple) and len(inp) > 0 else inp
        if isinstance(arg, dict):
            return _proxy_dict_obs(mod, inp, arg, learner)
        if torch.is_tensor(arg) and arg.dim() == 2 and arg.shape[1] == 150:
            return _proxy_tensor_obs(mod, inp, arg, learner)
        return None
    except (KeyError, TypeError):
        return None


def _proxy_dict_obs(
    mod: torch.nn.Module,
    inp: tuple[object, ...] | object,
    arg: dict[str, object],
    learner: object,
) -> tuple[object, ...] | dict[str, object] | None:
    t = arg.get("obs", None)
    if not (torch.is_tensor(t) and t.dim() == 2 and t.shape[1] == 150):
        return None
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1e6, 1e6)
    x = t.detach().requires_grad_(True)
    gt = getattr(learner, "_grad_tracker", None)
    if gt:
        x.register_hook(lambda g: gt.consume_grad(g))
    mod._obs_proxy = x  # type: ignore[attr-defined]
    new_arg = dict(arg)
    new_arg["obs"] = x
    if isinstance(inp, tuple):
        return (new_arg,) + tuple(inp[1:])  # type: ignore[index]
    return new_arg


def _proxy_tensor_obs(
    mod: torch.nn.Module,
    inp: tuple[object, ...] | object,
    arg: torch.Tensor,
    learner: object,
) -> tuple[object, ...] | torch.Tensor:
    t = torch.nan_to_num(arg, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1e6, 1e6)
    x = t.detach().requires_grad_(True)
    gt = getattr(learner, "_grad_tracker", None)
    if gt:
        x.register_hook(lambda g: gt.consume_grad(g))
    mod._obs_proxy = x  # type: ignore[attr-defined]
    if isinstance(inp, tuple):
        lst = list(inp)
        lst[0] = x
        return tuple(lst)
    return x


def _emit_initial_curriculum_keys(learner: object) -> None:
    """Emit initial curriculum keys to W&B at step 0."""
    import wandb

    frames0 = int(learner.train_step)  # type: ignore[attr-defined]
    boot: dict[str, float | int] = {"frames": frames0}
    for k in CURRICULUM_KEYS:
        boot[f"episode_extra_stats/{k}"] = 0.0
    wandb.log(boot, step=frames0)


def create_enhanced_train(
    original_train: Callable[..., object],
    cfg: Config,
    last_curriculum: dict[str, float],
) -> Callable[..., object]:
    """Return a patched Learner.train that logs curriculum and obs-grad metrics."""

    def enhanced_train(self: object, *args: object, **kwargs: object) -> object:
        result = original_train(self, *args, **kwargs)
        current_step = self.train_step  # type: ignore[attr-defined]

        try:
            import wandb

            frames = int(current_step)
            latest = getattr(self, "last_episodic_stats", None)
            if isinstance(latest, dict):
                _log_curriculum_metrics(latest, frames, last_curriculum)
                _log_trajectory_metrics(latest, frames)
        except RuntimeError:
            pass

        _step_influence_tracker(self, current_step, cfg)
        _step_grad_tracker(self, current_step, cfg)
        _explicit_obs_grad_log(self, current_step)

        return result

    return enhanced_train


def _log_curriculum_metrics(
    latest: dict[str, object],
    frames: int,
    last_curriculum: dict[str, float],
) -> None:
    """Extract curriculum metrics from episodic stats and log to W&B."""
    import wandb

    try:
        cur_payload: dict[str, float | int] = {}
        for k in CURRICULUM_KEYS:
            v = _get_with_prefixes(latest, k)
            if v is not None:
                cur_payload[f"episode_extra_stats/{k}"] = v

        ts = cur_payload.get("episode_extra_stats/curriculum/total_successes")
        tc = cur_payload.get("episode_extra_stats/curriculum/total_crashes")
        tt = cur_payload.get("episode_extra_stats/curriculum/total_timeouts")
        if ts is not None and tc is not None and tt is not None:
            cur_payload["episode_extra_stats/curriculum/total_resets"] = float(
                ts + tc + tt  # type: ignore[operator]
            )

        for k in list(cur_payload.keys()):
            if isinstance(k, str) and (
                k.endswith("curriculum/current_level")
                or k.endswith("curriculum/current_level_minus_1")
                or k.endswith("curriculum/current_progress")
            ):
                del cur_payload[k]

        if cur_payload:
            cur_payload["frames"] = frames
            wandb.log(cur_payload, step=frames)

        for k in CURRICULUM_KEYS:
            ep_key = f"episode_extra_stats/{k}"
            if ep_key in cur_payload:
                last_curriculum[k] = float(cur_payload[ep_key])

        forward_payload: dict[str, float | int] = {"frames": frames}
        for k in CURRICULUM_KEYS:
            if k in (
                "curriculum/current_level",
                "curriculum/current_level_minus_1",
                "curriculum/current_progress",
            ):
                continue
            forward_payload[f"episode_extra_stats/{k}"] = float(last_curriculum.get(k, 0.0))
        if len(forward_payload) > 1:
            wandb.log(forward_payload, step=frames)
    except RuntimeError:
        pass


def _get_with_prefixes(latest: dict[str, object], key_name: str) -> float | None:
    """Try multiple namespaces to find the latest value."""
    v = get_last_stat(latest, key_name)
    if v is not None:
        return v
    v = get_last_stat(latest, "episode_extra_stats/" + key_name)
    if v is not None:
        return v
    if key_name.startswith("curriculum/"):
        bare = key_name.split("/", 1)[1]
        return get_last_stat(latest, bare)
    return None


def _log_trajectory_metrics(latest: dict[str, object], frames: int) -> None:
    """Log trajectory and VAE diagnostics from episodic stats."""
    import wandb

    traj_payload: dict[str, float | int] = {}

    _TRAJ_KEYS: list[tuple[str, list[str]]] = [
        (
            "episode_extra_stats/path_efficiency",
            [
                "episode_extra_stats/path_efficiency_running_mean",
                "path_efficiency_running_mean",
                "episode_extra_stats/path_efficiency",
            ],
        ),
        (
            "episode_extra_stats/time_to_gate_steps",
            [
                "episode_extra_stats/time_to_gate_running_mean",
                "time_to_gate_running_mean",
                "episode_extra_stats/time_to_gate_steps",
            ],
        ),
        (
            "episode_extra_stats/min_gate_distance",
            [
                "episode_extra_stats/min_gate_distance_running_mean",
                "min_gate_distance_running_mean",
                "episode_extra_stats/min_gate_distance",
            ],
        ),
        (
            "episode_extra_stats/center_offset_success",
            [
                "episode_extra_stats/center_offset_running_mean",
                "center_offset_running_mean",
                "episode_extra_stats/center_offset_success",
            ],
        ),
        (
            "episode_extra_stats/height_offset_success",
            [
                "episode_extra_stats/height_offset_running_mean",
                "height_offset_running_mean",
                "episode_extra_stats/height_offset_success",
            ],
        ),
    ]

    for out_key, search_keys in _TRAJ_KEYS:
        val = get_last_any(latest, search_keys)
        if val is not None:
            if "time_to_gate" in out_key and not math.isfinite(val):
                continue
            traj_payload[out_key] = val

    for simple_key in (
        "episode_extra_stats/gate_pass_rate",
        "episode_extra_stats/episodes_total",
        "episode_extra_stats/episodes_crossed",
    ):
        val = get_last_stat(latest, simple_key)
        if val is not None:
            traj_payload[simple_key] = val

    _VAE_KEYS = (
        "episode_extra_stats/vae/drone_mean",
        "episode_extra_stats/vae/static_mean",
        "episode_extra_stats/vae/drone_std",
        "episode_extra_stats/vae/static_std",
        "episode_extra_stats/vae/drone_dim_std_mean",
        "episode_extra_stats/vae/static_dim_std_mean",
        "episode_extra_stats/vae/static_to_drone_norm_ratio",
    )
    for name in _VAE_KEYS:
        v = get_last_any(latest, [name, name.replace("episode_extra_stats/", "")])
        if v is not None:
            traj_payload[name] = float(v)

    if traj_payload:
        traj_payload["frames"] = frames
        wandb.log(traj_payload, step=frames)


def _step_influence_tracker(learner: object, current_step: int, cfg: Config) -> None:
    tracker = getattr(learner, "_influence_tracker", None)
    if tracker and tracker.enabled:
        tracker.step_count = current_step
        if current_step > 0 and current_step % cfg.gradient_print_interval == 0:
            tracker.print_analysis_summary()


def _step_grad_tracker(learner: object, current_step: int, cfg: Config) -> None:
    tracker = getattr(learner, "_grad_tracker", None)
    if tracker and tracker.enabled:
        if current_step > 0 and current_step % cfg.gradient_print_interval == 0:
            tracker.print_gradient_summary()


def _explicit_obs_grad_log(learner: object, current_step: int) -> None:
    """Explicitly log obs-grad metrics to W&B under episode_extra_stats."""
    try:
        import wandb

        frames = int(current_step)
        metric_sources: list[dict[str, object]] = []
        inf_tracker = getattr(learner, "_influence_tracker", None)
        if inf_tracker and inf_tracker.enabled:
            metric_sources.append(inf_tracker.get_logging_metrics())
        grd_tracker = getattr(learner, "_grad_tracker", None)
        if grd_tracker and grd_tracker.enabled:
            metric_sources.append(grd_tracker.get_logging_metrics())

        merged = metrics_to_float({k: v for src in metric_sources for k, v in src.items()})

        obs_payload: dict[str, float | int] = {}
        for name, val in merged.items():
            if is_obs_grad_key(name):
                key_tail = name.split("/", 1)[1] if "/" in name else name
                obs_payload["episode_extra_stats/obs_grad/" + key_tail] = val

        shares = compute_windowed_shares(merged)
        for k, v in shares.items():
            obs_payload["episode_extra_stats/obs_grad/" + k] = v

        if obs_payload:
            obs_payload["frames"] = frames
            wandb.log(obs_payload, step=frames)
    except RuntimeError:
        pass
