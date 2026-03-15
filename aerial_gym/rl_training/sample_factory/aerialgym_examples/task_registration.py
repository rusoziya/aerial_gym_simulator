"""Task registration helpers for Sample Factory gate navigation training."""

from __future__ import annotations

import os

from sample_factory.utils.typing import Config

from aerial_gym.registry.task_registry import task_registry


def register_gate_task_in_subprocess(
    full_task_name: str,
    cfg: Config,
) -> None:
    """Ensure the DCE navigation task is registered in a Sample Factory worker subprocess."""
    if full_task_name not in ("quad_with_obstacles", "quad_with_obstacles_gate"):
        return

    try:
        task_registry.get_task_class(full_task_name)
        return
    except KeyError:
        pass

    try:
        if full_task_name == "quad_with_obstacles_gate":
            _register_gate_variant(cfg)
        else:
            _register_standard_variant(cfg)
    except (ValueError, TypeError) as e:
        print(f"Failed to register {full_task_name} in subprocess: {e}")


def _register_gate_variant(cfg: Config) -> None:
    from aerial_gym.examples.dce_rl_navigation.dce_navigation_task_gate import (
        DCE_RL_Navigation_Task_Gate,
    )
    from aerial_gym.config.task_config.navigation_task_config_gate import task_config

    gate_config = task_config()
    _apply_ablation_flags(gate_config, cfg)
    _apply_subprocess_env_settings(gate_config, cfg)

    gate_config.headless = True

    cap = _safe_attr(cfg, "max_curriculum_level", None)
    if cap is not None:
        try:
            gate_config.max_curriculum_level = int(cap)
        except (ValueError, TypeError):
            gate_config.max_curriculum_level = None

    task_registry.register_task(
        "quad_with_obstacles_gate", DCE_RL_Navigation_Task_Gate, gate_config
    )
    task_registry.register_task(
        "dce_navigation_task_gate", DCE_RL_Navigation_Task_Gate, gate_config
    )


def _register_standard_variant(cfg: Config) -> None:
    from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import (
        DCE_RL_Navigation_Task,
    )

    base_config = task_registry.get_task_config("navigation_task")
    config = base_config()
    config.action_space_dim = 3
    config.curriculum.min_level = 3
    config.curriculum.max_level = 23
    config.headless = True

    cap = _safe_attr(cfg, "max_curriculum_level", None)
    if cap is not None:
        try:
            config.max_curriculum_level = int(cap)
        except (ValueError, TypeError):
            config.max_curriculum_level = None

    _apply_subprocess_env_settings(config, cfg)

    task_registry.register_task("quad_with_obstacles", DCE_RL_Navigation_Task, config)
    task_registry.register_task("dce_navigation_task", DCE_RL_Navigation_Task, config)


def register_main_process_tasks() -> None:
    """Register both standard and gate DCE tasks in the main process."""
    _register_main_standard_task()
    _register_main_gate_task()


def _register_main_standard_task() -> None:
    try:
        from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import (
            DCE_RL_Navigation_Task,
        )

        base_config = task_registry.get_task_config("navigation_task")
        dce_config = base_config()
        dce_config.action_space_dim = 3
        dce_config.curriculum.min_level = 3
        dce_config.curriculum.max_level = 23

        task_registry.register_task("quad_with_obstacles", DCE_RL_Navigation_Task, dce_config)
        task_registry.register_task("dce_navigation_task", DCE_RL_Navigation_Task, dce_config)
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not register quad_with_obstacles: {e}")


def _register_main_gate_task() -> None:
    try:
        from aerial_gym.examples.dce_rl_navigation.dce_navigation_task_gate import (
            DCE_RL_Navigation_Task_Gate,
        )
        from aerial_gym.config.task_config.navigation_task_config_gate import (
            task_config as gate_task_config,
        )

        gate_config = gate_task_config()
        task_registry.register_task(
            "quad_with_obstacles_gate", DCE_RL_Navigation_Task_Gate, gate_config
        )
        task_registry.register_task(
            "dce_navigation_task_gate", DCE_RL_Navigation_Task_Gate, gate_config
        )
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not register quad_with_obstacles_gate: {e}")


def _apply_ablation_flags(gate_config: object, cfg: Config) -> None:
    """Apply gate-size and obstacle ablation flags from CLI config."""
    try:
        gate_config.disable_gate_size_randomization = bool(cfg.disable_gate_size_randomization)
    except (AttributeError, TypeError):
        gate_config.disable_gate_size_randomization = False

    try:
        gate_config.fixed_gate_scale_percent = int(cfg.fixed_gate_scale_percent)
    except (ValueError, TypeError):
        gate_config.fixed_gate_scale_percent = 100

    try:
        gate_config.disable_obstacle_randomization = bool(cfg.disable_obstacle_randomization)
    except (AttributeError, TypeError):
        gate_config.disable_obstacle_randomization = False

    try:
        gate_config.fixed_obstacles_behind_gate = int(cfg.fixed_obstacles_behind_gate)
    except (ValueError, TypeError):
        gate_config.fixed_obstacles_behind_gate = 0


def _apply_subprocess_env_settings(config: object, cfg: Config) -> None:
    """Apply environment count settings from CLI config."""
    env_agents = _safe_attr(cfg, "env_agents", 0)
    if env_agents and env_agents > 0:
        config.num_envs = env_agents
        os.environ["SF_ENV_AGENTS"] = str(env_agents)


def _safe_attr(obj: object, name: str, default: object) -> object:
    """Safely retrieve an attribute with a default, without using getattr on self."""
    try:
        return obj.__dict__.get(name, default)
    except AttributeError:
        # Config objects may not have __dict__ (e.g. Namespace)
        try:
            return vars(obj).get(name, default)
        except TypeError:
            return default
