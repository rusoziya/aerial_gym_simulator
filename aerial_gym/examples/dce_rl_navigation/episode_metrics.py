"""Running episode-level metrics and W&B logging for gate navigation inference."""

from __future__ import annotations

import time

import numpy as np
import torch


class EpisodeMetrics:
    """Tracks per-episode and running aggregate metrics across all environments."""

    def __init__(self, num_envs: int, num_actions: int) -> None:
        self.num_envs: int = num_envs
        self.frames: int = 0
        self.episodes_done: int = 0
        self.episode_counter_total: int = 0

        self.ep_return: torch.Tensor = torch.zeros(num_envs, dtype=torch.float32)
        self.ep_length: torch.Tensor = torch.zeros(num_envs, dtype=torch.int32)

        self.prev_actions: torch.Tensor = torch.zeros(num_envs, num_actions, dtype=torch.float32)
        self.ep_action_diff_sum: torch.Tensor = torch.zeros(
            num_envs, num_actions, dtype=torch.float32
        )
        self.ep_action_diff_count: torch.Tensor = torch.zeros(num_envs, dtype=torch.int32)

        self.hist_returns: list[float] = []
        self.hist_returns_success: list[float] = []
        self.hist_lengths: list[float] = []
        self.hist_lengths_success: list[float] = []
        self.hist_center_offset: list[float] = []
        self.hist_height_offset: list[float] = []
        self.hist_min_gate_distance: list[float] = []

        self.traj_running: dict[str, float | int] = {
            "path_efficiency_sum": 0.0,
            "path_efficiency_count": 0,
            "min_gate_distance_sum": 0.0,
            "min_gate_distance_count": 0,
            "time_to_gate_sum": 0.0,
            "time_to_gate_count": 0,
            "center_offset_sum": 0.0,
            "center_offset_count": 0,
            "height_offset_sum": 0.0,
            "height_offset_count": 0,
            "target_success_sum": 0.0,
            "target_success_count": 0,
            "gate_pass_rate_sum": 0.0,
            "gate_pass_rate_count": 0,
            "episodes_total": 0,
            "episodes_crossed": 0,
        }
        self.totals: dict[str, int] = {"successes": 0, "crashes": 0, "timeouts": 0}

        self.success_ema: float | None = None
        self._ema_alpha: float = 0.1
        self._t0: float = time.time()

    def step(self, reward: torch.Tensor | None, actions: torch.Tensor) -> ActionStats | None:
        """Update per-step accumulators. Returns action stats or None."""
        self.frames += 1
        if isinstance(reward, torch.Tensor):
            self.ep_return += reward.float().cpu()
        self.ep_length += 1
        try:
            a = actions.detach().float().cpu()
            adiff = torch.abs(a - self.prev_actions)
            abs_mean = torch.mean(torch.abs(a), dim=0)
            diff_mean = torch.mean(adiff, dim=0)
            sat = torch.mean((torch.abs(a) > 0.95).float()).item()
            self.prev_actions = a
            self.ep_action_diff_sum += adiff
            self.ep_action_diff_count += 1
            return ActionStats(abs_mean=abs_mean, diff_mean=diff_mean, saturation=sat)
        except (ValueError, TypeError):
            return None

    def on_resets(
        self,
        ids: torch.Tensor,
        infos: dict[str, object],
        nn_model: object,
    ) -> dict[str, float]:
        """Process environment resets. Returns W&B payload dict."""
        ids_cpu = ids.cpu()
        ret = self.ep_return[ids_cpu].numpy()
        length = self.ep_length[ids_cpu].numpy()

        self.ep_return[ids_cpu] = 0.0
        self.ep_length[ids_cpu] = 0
        nn_model.reset(ids)

        payload: dict[str, float] = {}
        self._update_episode_action_diffs(ids, ids_cpu, payload)

        succ_mask = self._extract_success_mask(ids, infos)
        self._update_histories(ret, length, succ_mask)
        self._add_return_length_stats(payload, ret, length)
        self._add_early_crash_rate(payload, ids, infos, length)

        self._update_trajectory_running(infos)
        self._add_running_means(payload)
        self._add_totals_and_rates(payload, infos)
        self._add_spatial_quantiles(payload)

        self.episodes_done += int(ids.numel())
        self.episode_counter_total += int(ids.numel())
        return payload

    def _update_episode_action_diffs(
        self,
        ids: torch.Tensor,
        ids_cpu: torch.Tensor,
        payload: dict[str, float],
    ) -> None:
        counts = self.ep_action_diff_count[ids].float().clamp_min(1.0)
        per_env_means = self.ep_action_diff_sum[ids] / counts.unsqueeze(1)
        diff_ep_mean = per_env_means.mean(dim=0)
        dims = ["x", "y", "z", "yaw"]
        for i, d in enumerate(dims[: diff_ep_mean.numel()]):
            payload[f"episode_action_diff_mean/{d}"] = float(diff_ep_mean[i].item())
        self.ep_action_diff_sum[ids] = 0.0
        self.ep_action_diff_count[ids] = 0

    def _extract_success_mask(
        self, ids: torch.Tensor, infos: dict[str, object]
    ) -> np.ndarray | None:
        try:
            s = infos.get("successes", None)
            if isinstance(s, torch.Tensor):
                return s[ids].bool().cpu().numpy()
        except (KeyError, TypeError):
            pass
        return None

    def _update_histories(
        self,
        ret: np.ndarray,
        length: np.ndarray,
        succ_mask: np.ndarray | None,
    ) -> None:
        self.hist_returns.extend(ret.tolist())
        self.hist_lengths.extend(length.tolist())
        if succ_mask is not None and succ_mask.size == ret.size:
            self.hist_returns_success.extend(ret[succ_mask].tolist())
            self.hist_lengths_success.extend(length[succ_mask].tolist())

    def _add_return_length_stats(
        self,
        payload: dict[str, float],
        ret: np.ndarray,
        length: np.ndarray,
    ) -> None:
        if len(self.hist_returns) > 0:
            payload["episode_return_mean"] = float(np.mean(self.hist_returns))
            payload["episode_return_std"] = float(np.std(self.hist_returns))
        if len(self.hist_lengths) > 0:
            payload["episode_length_mean"] = float(np.mean(self.hist_lengths))
        if len(self.hist_returns_success) > 0:
            payload["success_only/episode_return_mean"] = float(np.mean(self.hist_returns_success))
        if len(self.hist_lengths_success) > 0:
            payload["success_only/episode_length_mean"] = float(np.mean(self.hist_lengths_success))

    def _add_early_crash_rate(
        self,
        payload: dict[str, float],
        ids: torch.Tensor,
        infos: dict[str, object],
        length: np.ndarray,
    ) -> None:
        c = infos.get("crashes", None)
        if isinstance(c, torch.Tensor):
            crashed = c[ids].bool().cpu().numpy()
            early = length <= 5
            denom = max(1, crashed.size)
            payload["early_crash_rate"] = float((crashed & early).sum() / denom)

    def _update_trajectory_running(self, infos: dict[str, object]) -> None:
        if not isinstance(infos, dict):
            return
        extra = infos.get("episode_extra_stats", None)
        if not isinstance(extra, dict):
            return
        _KEYS = [
            ("path_efficiency", "path_efficiency"),
            ("time_to_gate_steps", "time_to_gate"),
            ("min_gate_distance", "min_gate_distance"),
            ("center_offset_success", "center_offset"),
            ("height_offset_success", "height_offset"),
            ("target_success_rate", "target_success"),
            ("gate_pass_rate", "gate_pass_rate"),
        ]
        for info_key, running_key in _KEYS:
            val = _to_float(extra.get(info_key))
            if val is not None:
                self.traj_running[f"{running_key}_sum"] += val
                self.traj_running[f"{running_key}_count"] += 1
        co = _to_float(extra.get("center_offset_success"))
        ho = _to_float(extra.get("height_offset_success"))
        mgd = _to_float(extra.get("min_gate_distance"))
        if co is not None:
            self.hist_center_offset.append(co)
        if ho is not None:
            self.hist_height_offset.append(ho)
        if mgd is not None:
            self.hist_min_gate_distance.append(mgd)
        crossed = _to_float(extra.get("episodes_crossed"))
        total = _to_float(extra.get("episodes_total"))
        if total is not None:
            self.traj_running["episodes_total"] += int(total)
        elif crossed is not None:
            self.traj_running["episodes_total"] += 1
        if crossed is not None and crossed > 0:
            self.traj_running["episodes_crossed"] += int(crossed)

    def _add_running_means(self, payload: dict[str, float]) -> None:
        _PAIRS = [
            ("path_efficiency", False),
            ("min_gate_distance", False),
            ("time_to_gate", True),
            ("center_offset", True),
            ("height_offset", True),
            ("target_success", False),
            ("gate_pass_rate", False),
        ]
        for key, none_if_zero in _PAIRS:
            s = self.traj_running.get(f"{key}_sum", 0.0)
            c = self.traj_running.get(f"{key}_count", 0)
            if c <= 0:
                if not none_if_zero:
                    payload[f"{key}_running_mean"] = float("nan")
            else:
                payload[f"{key}_running_mean"] = float(s / c)
        if self.traj_running["episodes_total"] > 0:
            payload["episode_extra_stats/gate_pass_rate"] = float(
                self.traj_running["episodes_crossed"]
            ) / float(self.traj_running["episodes_total"])

    def _add_totals_and_rates(self, payload: dict[str, float], infos: dict[str, object]) -> None:
        if isinstance(infos, dict):
            for key in ("successes", "crashes", "timeouts"):
                val = infos.get(key, None)
                if isinstance(val, torch.Tensor):
                    c = float(val.sum().item())
                    payload[f"episode_extra_stats/{key}"] = c
                    self.totals[key] += int(c)
                elif isinstance(val, (int, float)):
                    c = float(val)
                    payload[f"episode_extra_stats/{key}"] = c
                    self.totals[key] += int(c)

        total_resets = self.totals["successes"] + self.totals["crashes"] + self.totals["timeouts"]
        payload["episode_extra_stats/curriculum/total_successes"] = float(self.totals["successes"])
        payload["episode_extra_stats/curriculum/total_crashes"] = float(self.totals["crashes"])
        payload["episode_extra_stats/curriculum/total_timeouts"] = float(self.totals["timeouts"])
        payload["episode_extra_stats/curriculum/total_resets"] = float(total_resets)
        payload["episodes"] = float(total_resets)
        if total_resets > 0:
            sr = float(self.totals["successes"]) / float(total_resets)
            payload["episode_extra_stats/curriculum/success_rate"] = sr
            payload["episode_extra_stats/curriculum/crash_rate"] = float(
                self.totals["crashes"]
            ) / float(total_resets)
            payload["episode_extra_stats/curriculum/timeout_rate"] = float(
                self.totals["timeouts"]
            ) / float(total_resets)
            if self.success_ema is None:
                self.success_ema = sr
            else:
                self.success_ema = (1 - self._ema_alpha) * self.success_ema + self._ema_alpha * sr
        if self.success_ema is not None:
            payload["success_rate_running"] = float(self.success_ema)

    def _add_spatial_quantiles(self, payload: dict[str, float]) -> None:
        if len(self.hist_center_offset) > 0:
            payload["spatial/center_offset_p50"] = float(np.percentile(self.hist_center_offset, 50))
            payload["spatial/center_offset_p90"] = float(np.percentile(self.hist_center_offset, 90))
        if len(self.hist_height_offset) > 0:
            payload["spatial/height_offset_p50"] = float(np.percentile(self.hist_height_offset, 50))
            payload["spatial/height_offset_p90"] = float(np.percentile(self.hist_height_offset, 90))
        if len(self.hist_min_gate_distance) > 0:
            payload["spatial/min_gate_distance_p50"] = float(
                np.percentile(self.hist_min_gate_distance, 50)
            )
            payload["spatial/min_gate_distance_p90"] = float(
                np.percentile(self.hist_min_gate_distance, 90)
            )

    def add_action_stats(self, payload: dict[str, float], action_stats: ActionStats | None) -> None:
        """Add action-level stats to the W&B payload."""
        if action_stats is None:
            return
        dims = ["x", "y", "z", "yaw"]
        if action_stats.abs_mean is not None:
            for i, d in enumerate(dims[: action_stats.abs_mean.numel()]):
                payload[f"action_abs_mean/{d}"] = float(action_stats.abs_mean[i].item())
        if action_stats.diff_mean is not None:
            for i, d in enumerate(dims[: action_stats.diff_mean.numel()]):
                payload[f"action_diff_mean/{d}"] = float(action_stats.diff_mean[i].item())
        if action_stats.saturation is not None:
            payload["action_saturation_rate"] = float(action_stats.saturation)

    def add_vae_and_throughput(
        self,
        payload: dict[str, float],
        vae_drone_norm: float | None,
        vae_static_norm: float | None,
        vae_ratio: float | None,
    ) -> None:
        """Add VAE norms and throughput stats to the W&B payload."""
        if vae_drone_norm is not None:
            payload["vae/drone_norm_mean"] = vae_drone_norm
            payload["vae/static_norm_mean"] = vae_static_norm if vae_static_norm else 0.0
            payload["vae/static_to_drone_norm_ratio"] = vae_ratio if vae_ratio else 0.0
        dt = max(1e-6, time.time() - self._t0)
        payload["throughput/fps_env"] = float(self.frames / dt)
        payload["throughput/episodes_per_min"] = float((self.episode_counter_total / dt) * 60.0)
        payload["global_step"] = self.frames
        payload["frames"] = self.frames


class ActionStats:
    """Container for per-step action statistics."""

    def __init__(
        self,
        abs_mean: torch.Tensor | None,
        diff_mean: torch.Tensor | None,
        saturation: float | None,
    ) -> None:
        self.abs_mean: torch.Tensor | None = abs_mean
        self.diff_mean: torch.Tensor | None = diff_mean
        self.saturation: float | None = saturation


class VisibilityTracker:
    """Tracks per-env per-episode visibility/FOV metrics and running aggregates."""

    _FIELDS: tuple[str, ...] = (
        "abs",
        "frustum",
        "eff",
        "fov_score",
        "fov_visible",
        "fov_horiz",
        "fov_vert",
    )
    _INFO_KEYS: dict[str, str] = {
        "abs": "static_visibility/abs",
        "frustum": "static_visibility/frustum",
        "eff": "static_visibility/eff",
        "fov_score": "static_fov/score",
        "fov_visible": "static_fov/visible",
        "fov_horiz": "static_fov/horiz_angle_rad",
        "fov_vert": "static_fov/vert_angle_rad",
    }

    def __init__(self, num_envs: int) -> None:
        self._sum_env: dict[str, torch.Tensor] = {}
        self._count_env: dict[str, torch.Tensor] = {}
        self._running_sum: dict[str, float] = {}
        self._running_count: dict[str, int] = {}
        for f in self._FIELDS:
            self._sum_env[f] = torch.zeros(num_envs, dtype=torch.float32)
            self._count_env[f] = torch.zeros(num_envs, dtype=torch.int32)
            self._running_sum[f] = 0.0
            self._running_count[f] = 0

    def accumulate(self, infos: dict[str, object]) -> None:
        """Add per-step visibility values from infos dict."""
        if not isinstance(infos, dict):
            return
        for field, info_key in self._INFO_KEYS.items():
            val = infos.get(info_key)
            if isinstance(val, torch.Tensor):
                self._sum_env[field] += val.detach().float().cpu()
                self._count_env[field] += 1

    def on_resets(self, ids: torch.Tensor) -> None:
        """Finalize episode means for resetting envs and update running aggregates."""
        ids_cpu = ids.cpu()
        for f in self._FIELDS:
            s = self._sum_env[f][ids_cpu]
            c = self._count_env[f][ids_cpu].float().clamp_min(1.0)
            means = (s / c).numpy()
            self._running_sum[f] += float(means.sum())
            self._running_count[f] += int(means.size)
            self._sum_env[f][ids_cpu] = 0.0
            self._count_env[f][ids_cpu] = 0

    def get_running_payload(self, frames: int, total_resets: int) -> dict[str, float]:
        """Build a W&B payload dict with running-mean visibility/FOV metrics."""
        payload: dict[str, float] = {
            "frames": frames,
            "global_step": frames,
            "episodes": float(total_resets),
        }
        _METRIC_NAMES = {
            "abs": "visibility/abs_running_mean",
            "frustum": "visibility/frustum_running_mean",
            "eff": "visibility/eff_running_mean",
            "fov_score": "fov/score_running_mean",
            "fov_visible": "fov/visible_rate_running_mean",
            "fov_horiz": "fov/horiz_angle_rad_running_mean",
            "fov_vert": "fov/vert_angle_rad_running_mean",
        }
        for f, metric in _METRIC_NAMES.items():
            c = max(1, self._running_count[f])
            payload[metric] = float(self._running_sum[f] / c)
        return payload


def _to_float(val: object) -> float | None:
    """Safely convert a scalar/tensor to float, returning None on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        if isinstance(val, torch.Tensor) and val.numel() == 1:
            return float(val.item())
        return None
