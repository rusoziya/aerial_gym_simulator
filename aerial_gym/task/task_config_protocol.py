"""Protocol definition for task configuration objects.

All task config classes (navigation_task_config, navigation_task_config_gate, etc.)
conform to this protocol. Using a Protocol instead of Any gives full type safety
without requiring config classes to explicitly inherit from a base.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Callable

import torch


@runtime_checkable
class CurriculumConfig(Protocol):
    """Protocol for the nested curriculum config class."""

    min_level: int
    max_level: int
    check_after_log_instances: int
    increase_step: int
    decrease_step: int
    success_rate_for_increase: float
    success_rate_for_decrease: float


@runtime_checkable
class VaeConfig(Protocol):
    """Protocol for the nested VAE config class."""

    use_vae: bool
    latent_dims: int
    model_file: str
    image_res: tuple[int, int]


@runtime_checkable
class TaskConfig(Protocol):
    """Protocol for task configuration objects.

    Any config class that defines these attributes satisfies this protocol.
    """

    seed: int
    sim_name: str
    env_name: str
    robot_name: str
    controller_name: str
    args: dict[str, str]
    num_envs: int
    use_warp: bool
    headless: bool
    device: str
    observation_space_dim: int
    privileged_observation_space_dim: int
    action_space_dim: int
    episode_len_steps: int
    return_state_before_reset: bool
    reward_parameters: dict[str, torch.Tensor]
    action_transformation_function: Callable[[torch.Tensor], torch.Tensor]
    vae_config: VaeConfig
    curriculum: CurriculumConfig
