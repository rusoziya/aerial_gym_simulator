"""Training script for DCE navigation task (3D actions, 81D observations).

Uses the base AerialGymVecEnv wrapper for standard position-setpoint and
DCE navigation tasks. Compatible with existing inference scripts.
"""

from __future__ import annotations

import sys
import os

import isaacgym  # noqa: F401 — must import before torch
import gymnasium as gym
import torch
import numpy as np

from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.utils.typing import Config, Env
from sample_factory.utils.utils import str2bool

from aerial_gym.registry.task_registry import task_registry

from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_common import (
    AerialGymVecEnv,
    BASE_ENV_CONFIGS,
    override_default_params,
    clear_sf_cache,
    setup_env_agents,
    parse_cfg,
)


# Environment factory


def make_aerialgym_env(
    full_task_name: str,
    cfg: Config,
    _env_config: object = None,
    render_mode: str | None = None,
) -> Env:
    """Create an AerialGymVecEnv for Sample Factory."""
    task_name = cfg.env
    env_agents = cfg.env_agents if "env_agents" in cfg else 16

    # Update env agents for worker processes
    os.environ["SF_ENV_AGENTS"] = str(env_agents)
    os.environ["SF_HEADLESS"] = str(cfg.headless if "headless" in cfg else True)

    task = task_registry.make_task(task_name, num_envs=env_agents)
    env = AerialGymVecEnv(task, obs_key="obs", action_dim=3, obs_dim=81)

    print(f"[make_aerialgym_env] Created env: {task_name}, agents={env_agents}, "
          f"obs={env.observation_space}, act={env.action_space}")
    return env


# CLI parameters


def add_extra_params_func(parser: object) -> None:
    """Register custom CLI arguments for DCE navigation training."""
    p = parser
    p.add_argument("--env_agents", default=16, type=int,
                   help="Number of environments (agents) to create")
    p.add_argument("--headless", default=True, type=str2bool,
                   help="Run in headless mode (no visualization)")
    p.add_argument("--use_rnn", default=False, type=str2bool,
                   help="Use RNN-based policy")


def override_default_params_func(env: str, parser: object) -> None:
    """Apply default hyperparameters for this trainer."""
    override_default_params(env, parser, env_configs)


# Per-environment config overrides

env_configs: dict[str, dict[str, object]] = {
    **BASE_ENV_CONFIGS,
    "quad_with_obstacles": {
        "encoder_type": "mlp",
        "encoder_subtype": "mlp_mujoco",
        "encoder_extra_fc_layers": 0,
        "hidden_size": 64,
        "use_rnn": True,
        "rnn_size": 64,
        "rnn_type": "lstm",
        "rnn_num_layers": 1,
        "env_agents": 16,
        "batch_size": 8192,
        "num_batches_per_epoch": 4,
        "train_for_env_steps": 100000000,
    },
}


# Component registration


def register_aerialgym_custom_components() -> None:
    """Register DCE navigation task and environments with Sample Factory."""
    clear_sf_cache()
    setup_env_agents()

    try:
        from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import (
            DCE_RL_Navigation_Task,
        )
        from aerial_gym.config.task_config.navigation_task_config import task_config

        base_config = task_registry.get_task_config("navigation_task")
        dce_config = base_config()
        dce_config.action_space_dim = 3
        dce_config.curriculum.min_level = 36

        task_registry.register_task(
            "quad_with_obstacles", DCE_RL_Navigation_Task, dce_config
        )
        task_registry.register_task(
            "dce_navigation_task", DCE_RL_Navigation_Task, dce_config
        )
        print("Registered quad_with_obstacles + dce_navigation_task")
    except RuntimeError as e:
        print(f"Warning: Could not register quad_with_obstacles: {e}")

    for env_name in env_configs:
        register_env(env_name, make_aerialgym_env)


# Entry point


def parse_aerialgym_cfg(evaluation: bool = False) -> Config:
    return parse_cfg(add_extra_params_func, override_default_params_func, evaluation)


def main() -> int:
    """Script entry point."""
    register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
