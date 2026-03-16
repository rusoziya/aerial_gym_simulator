"""Shared training infrastructure for Sample Factory aerial gym environments.

This module contains the base environment wrapper, default configuration,
and registration logic shared between the standard DCE navigation trainer
and the gate navigation trainer.
"""

from __future__ import annotations

import glob
import os
from typing import Callable

import gymnasium as gym

if os.environ.get("AERIAL_GYM_BACKEND", "isaacgym").lower() != "isaaclab":
    import isaacgym  # noqa: F401 — must import before torch
import numpy as np
import torch
from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.utils.typing import Config
from torch import Tensor

# Base environment wrapper


class AerialGymVecEnv(gym.Env):
    """Base wrapper for aerial gym environments in Sample Factory.

    Handles observation transformation from multi-dict to single "obs" key,
    and enforces a fixed action space dimensionality.

    Subclass and override _transform_obs / _transform_action for variants.
    """

    def __init__(
        self,
        aerialgym_env: object,
        obs_key: str,
        action_dim: int = 3,
        obs_dim: int = 81,
    ) -> None:
        self.env = aerialgym_env
        self.num_agents: int = self.env.num_envs
        self.is_multiagent: bool = True

        base_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        self.action_space = convert_space(base_action_space)

        self.observation_space = gym.spaces.Dict(
            {
                "obs": convert_space(
                    gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                )
            }
        )

        self._truncated: Tensor = torch.zeros(self.num_agents, dtype=torch.bool)

    def _transform_obs(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Transform task observations to Sample Factory format."""
        return {"obs": obs["observations"]}

    def _transform_action(self, action: Tensor) -> Tensor:
        """Transform Sample Factory actions before passing to task."""
        return action

    def reset(self, *args: object, **kwargs: object) -> tuple[dict[str, Tensor], dict[str, object]]:
        obs, _rew, _terminated, _truncated, infos = self.env.reset()
        return self._transform_obs(obs), infos

    def step(
        self, action: Tensor
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, object]]:
        transformed_action = self._transform_action(action)
        obs, rew, terminated, truncated, infos = self.env.step(transformed_action)
        return self._transform_obs(obs), rew, terminated, truncated, infos

    def render(self) -> None:
        pass


# Shared default hyperparameters


def override_default_params(
    env: str,
    parser: object,
    env_configs: dict[str, dict[str, object]],
) -> None:
    """Set default training hyperparameters for Sample Factory."""
    default_batch_size = 1024
    default_num_batches_per_epoch = 4

    parser.set_defaults(
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        actor_worker_gpus=[0],
        train_for_env_steps=10000000,
        use_rnn=False,
        adaptive_stddev=True,
        policy_initialization="torch_default",
        env_gpu_actions=True,
        env_gpu_observations=True,
        reward_scale=0.1,
        rollout=32,
        max_grad_norm=1.0,
        batch_size=default_batch_size,
        num_batches_per_epoch=default_num_batches_per_epoch,
        num_epochs=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=2.0,
        exploration_loss_coeff=0.001,
        nonlinearity="elu",
        learning_rate=3e-4,
        lr_schedule="kl_adaptive_epoch",
        lr_schedule_kl_threshold=0.016,
        shuffle_minibatches=False,
        gamma=0.98,
        gae_lambda=0.95,
        with_vtrace=False,
        value_bootstrap=True,
        normalize_input=True,
        normalize_returns=True,
        save_best_after=int(1e6),
        serial_mode=True,
        async_rl=True,
        use_env_info_cache=False,
        kl_loss_coeff=0.1,
        restart_behavior="resume",
    )

    if env in env_configs:
        parser.set_defaults(**env_configs[env])


# Shared env_configs (position_setpoint, navigation baseline)

BASE_ENV_CONFIGS: dict[str, dict[str, object]] = {
    "position_setpoint_task": {
        "encoder_type": "mlp",
        "encoder_subtype": "mlp_mujoco",
        "encoder_extra_fc_layers": 0,
        "hidden_size": 256,
        "env_agents": 16,
        "batch_size": 8192,
        "num_batches_per_epoch": 4,
        "train_for_env_steps": 50000000,
    },
    "navigation_task": {
        "encoder_type": "mlp",
        "encoder_subtype": "mlp_mujoco",
        "encoder_extra_fc_layers": 0,
        "hidden_size": 256,
        "env_agents": 16,
        "batch_size": 8192,
        "num_batches_per_epoch": 4,
        "train_for_env_steps": 100000000,
    },
}


# Shared registration helpers


def clear_sf_cache(pattern: str = "quad_with_obstacles") -> None:
    """Remove Sample Factory environment info cache files."""
    cache_files = glob.glob(f"/tmp/sf2_*/env_info_{pattern}*")
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
        except OSError:
            pass


def setup_env_agents() -> str:
    """Read and propagate SF_ENV_AGENTS environment variable."""
    current_env_agents = os.environ.get("SF_ENV_AGENTS", "16")
    os.environ["SF_ENV_AGENTS"] = current_env_agents
    return current_env_agents


def parse_cfg(
    add_extra_params_func: Callable[[object], None],
    override_func: Callable[[str, object], None],
    evaluation: bool = False,
) -> Config:
    """Parse Sample Factory configuration with custom params."""
    parser, partial_cfg = parse_sf_args(evaluation=evaluation)
    add_extra_params_func(parser)
    override_func(partial_cfg.env, parser)
    return parse_full_cfg(parser)
