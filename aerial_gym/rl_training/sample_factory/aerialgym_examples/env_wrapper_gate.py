"""Gate navigation environment wrapper for Sample Factory."""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from sample_factory.algo.utils.gymnasium_utils import convert_space
from torch import Tensor

from aerial_gym.rl_training.sample_factory.aerialgym_examples.episode_stats import (
    CurriculumCounters,
    TrajectoryAggregator,
    inject_curriculum_current_mirror,
    inject_curriculum_snapshot,
    inject_gate_camera_stats,
    inject_traj_and_level_stats,
)
from aerial_gym.rl_training.sample_factory.aerialgym_examples.gif_recorder import GifRecorder
from aerial_gym.rl_training.sample_factory.aerialgym_examples.obs_ablation import (
    OBS_SLICES,
    ObsAblation,
)
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_common import (
    AerialGymVecEnv as AerialGymVecEnvBase,
)
from aerial_gym.utils.env_flag_utils import read_env_bool
from aerial_gym.utils.tensor_utils import has_invalid, sanitize_tensor

VERBOSE = read_env_bool("TRAIN_VERBOSE", default=False)


class AerialGymVecEnvGate(AerialGymVecEnvBase):
    """Wrapper for isaacgym environments compatible with Sample Factory.

    Supports dual camera GIF saving and 4D action space for gate navigation.
    """

    def __init__(
        self,
        aerialgym_env: object,
        obs_key: str,
        save_gifs: bool = False,
    ) -> None:
        obs_dim = 150
        action_dim = 4
        super().__init__(aerialgym_env, obs_key, action_dim=action_dim, obs_dim=obs_dim)

        self.save_gifs = save_gifs
        self._gif_recorder: GifRecorder | None = None
        if self.save_gifs:
            if VERBOSE:
                print("[AerialGymVecEnv] GIF saving ENABLED for dual cameras (drone + static)")
            self._gif_recorder = GifRecorder(self.num_agents)

        self.step_count: int = 0
        self.action_space = convert_space(
            gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        )

        if VERBOSE:
            print(f"[AerialGymVecEnv] Forced action space shape: {self.action_space.shape}")
            print(
                f"[AerialGymVecEnv] is_multiagent: {self.is_multiagent}, "
                f"num_agents: {self.num_agents}"
            )

        self.fusion_mode: str = os.environ.get("SF_FUSION_MODE", "concat")
        self.gate_per_feature: bool = os.environ.get("SF_GATE_PER_FEATURE", "1") == "1"

        if obs_key != "obs":
            raise ValueError(f"Unknown observation key: {obs_key}")

        task_obs_dim: int = self.env.task_config.observation_space_dim
        if VERBOSE:
            print(f"[AerialGymVecEnv] Detected observation space: {task_obs_dim}D")

        self.observation_space = gym.spaces.Dict(
            {
                "obs": convert_space(
                    gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(task_obs_dim,), dtype=np.float32
                    )
                )
            }
        )

        self._truncated: Tensor = torch.zeros(self.num_agents, dtype=torch.bool)
        self.episode_count: int = 0
        self.obs_slices: dict[str, tuple[int, int]] = OBS_SLICES
        self._obs_ablation = ObsAblation()
        self._traj_agg = TrajectoryAggregator()
        self._curriculum_counters = CurriculumCounters()

        if self.fusion_mode == "gated":
            self._gated_fuser = _build_gated_fuser(self.gate_per_feature)

    def reset(self, *args: object, **kwargs: object) -> tuple[dict[str, Tensor], dict[str, object]]:
        obs, rew, terminated, truncated, infos = self.env.reset()

        if self._gif_recorder is not None:
            for env_id in range(self.num_agents):
                self._gif_recorder.clear_frames(env_id)

        transformed_obs = self._transform_and_sanitize_obs(obs)

        if self._gif_recorder is not None:
            self._gif_recorder.collect_frames(obs, self.env)

        return transformed_obs, infos

    def step(
        self, action: Tensor
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, object]]:
        dce_action = _sanitize_action(action)
        obs, rew, terminated, truncated, infos = self.env.step(dce_action)

        if self._gif_recorder is not None:
            self._gif_recorder.collect_frames(obs, self.env)

        transformed_obs = self._transform_and_sanitize_obs(obs)

        early = self._handle_gif_episode_end(terminated, truncated, transformed_obs, rew, infos)
        if early is not None:
            return early

        self._inject_episode_extra_stats(infos, terminated, truncated)
        self.step_count += 1
        return transformed_obs, rew, terminated, truncated, infos

    def render(self) -> None:
        pass

    def _transform_and_sanitize_obs(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Transform raw obs into Sample Factory format, apply ablation, and sanitize."""
        transformed_obs = {"obs": obs["observations"]}
        transformed_obs["obs"] = self._obs_ablation.apply(transformed_obs["obs"])
        vec = transformed_obs["obs"]
        if isinstance(vec, torch.Tensor):
            if has_invalid(vec):
                if read_env_bool("ABLATE_DEBUG", default=False):
                    n_bad = int((~torch.isfinite(vec)).sum().item())
                    print(f"[SANITIZE] replacing {n_bad} non-finite obs values with 0")
            transformed_obs["obs"] = sanitize_tensor(vec)
        return transformed_obs

    def _handle_gif_episode_end(
        self,
        terminated: Tensor,
        truncated: Tensor,
        transformed_obs: dict[str, Tensor],
        rew: Tensor,
        infos: dict[str, object],
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, object]] | None:
        """Handle GIF saving on episode boundaries; returns early-return tuple or None."""
        if self._gif_recorder is None:
            return None
        if not (torch.any(terminated) or torch.any(truncated)):
            return None

        reset_ids = (terminated + truncated).nonzero(as_tuple=True)[0]
        if len(reset_ids) == 0 or 0 not in reset_ids:
            return None

        try:
            env_agents = int(os.environ.get("SF_ENV_AGENTS", "0"))
        except (ValueError, TypeError):
            env_agents = 0
        if env_agents != 16:
            self._gif_recorder.clear_frames(env_id=0)
            self.episode_count += 1
            return transformed_obs, rew, terminated, truncated, infos

        if self.episode_count % 5 == 0:
            level_suffix = _get_level_suffix(self.env)
            self._gif_recorder.save_episode_gifs(env_id=0, level_suffix=level_suffix)
        self._gif_recorder.clear_frames(env_id=0)
        self.episode_count += 1
        return None

    def _inject_episode_extra_stats(
        self,
        infos: dict[str, object],
        terminated: Tensor,
        truncated: Tensor,
    ) -> None:
        """Inject curriculum level, trajectory, and curriculum stats into infos for W&B."""
        try:
            if not isinstance(infos, dict):
                return
            extra: dict[str, float] = infos.get("episode_extra_stats", {})
            if not isinstance(extra, dict):
                extra = {}
            ids = (terminated + truncated).nonzero(as_tuple=True)[0]
            if ids.numel() == 0:
                return

            task = self.env
            curr_level: float | None = None
            if task is not None:
                curr_level = task.curriculum_level
                reset_ids = ids.detach().cpu().tolist()
                self._traj_agg.update(reset_ids, task)

            inject_traj_and_level_stats(extra, task, curr_level)
            self._traj_agg.inject_running_means(extra)
            self._curriculum_counters.update_and_inject(extra, infos, terminated, truncated)
            inject_curriculum_current_mirror(extra, infos, task)
            inject_gate_camera_stats(extra, infos)
            inject_curriculum_snapshot(extra, infos, task)

            # Inject success/crash/timeout rates from running totals
            tot_s = float(self._curriculum_counters.total_successes)
            tot_c = float(self._curriculum_counters.total_crashes)
            tot_t = float(self._curriculum_counters.total_timeouts)
            total = max(1.0, tot_s + tot_c + tot_t)
            extra.setdefault("curriculum/success_rate", tot_s / total)
            extra.setdefault("curriculum/crash_rate", tot_c / total)
            extra.setdefault("curriculum/timeout_rate", tot_t / total)

            infos["episode_extra_stats"] = extra
        except (ValueError, TypeError):
            pass


def _sanitize_action(action: Tensor) -> Tensor:
    """Replace NaN/Inf with 0 and clamp to [-1, 1]."""
    if isinstance(action, torch.Tensor) and has_invalid(action):
        if read_env_bool("ABLATE_DEBUG", default=False):
            n_bad = int((~torch.isfinite(action)).sum().item())
            print(f"[SANITIZE][action] replacing {n_bad} non-finite action values with 0")
        action = sanitize_tensor(action).clamp_(-1.0, 1.0)
    return action


def _get_level_suffix(env: object) -> str:
    """Build a curriculum level suffix string for GIF filenames."""
    try:
        return f"_L{int(env.curriculum_level):02d}"  # type: ignore[attr-defined]
    except (AttributeError, ValueError, TypeError):
        return ""


def _build_gated_fuser(gate_per_feature: bool) -> nn.Module:
    """Construct the dual-gated late fusion module."""

    class DualGatedLateFusion(nn.Module):
        def __init__(
            self,
            latent_dim: int = 64,
            gate_per_feature: bool = True,
        ) -> None:
            super().__init__()
            D = latent_dim
            self.ego_proj = nn.Sequential(
                nn.LayerNorm(D), nn.Linear(D, D), nn.ELU(), nn.LayerNorm(D)
            )
            self.static_proj = nn.Sequential(
                nn.LayerNorm(D), nn.Linear(D, D), nn.ELU(), nn.LayerNorm(D)
            )
            gate_out = D if gate_per_feature else 1
            self.gate = nn.Sequential(nn.Linear(2 * D, D), nn.ELU(), nn.Linear(D, gate_out))

        def forward(self, ego_latent: Tensor, static_latent: Tensor) -> tuple[Tensor, Tensor]:
            e = self.ego_proj(ego_latent)
            s = self.static_proj(static_latent)
            g = torch.sigmoid(self.gate(torch.cat([e, s], dim=-1)))
            if g.shape[-1] == 1:
                g = g.expand_as(e)
            z = g * s + (1 - g) * e
            return z, g

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DualGatedLateFusion(latent_dim=64, gate_per_feature=gate_per_feature).to(device)
