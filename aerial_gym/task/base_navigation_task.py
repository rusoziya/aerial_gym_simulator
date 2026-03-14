"""Base class for navigation tasks, shared by NavigationTask and NavigationTaskGate.

Provides common initialisation (config overrides, SimBuilder, reward params, VAE,
observation/action spaces, task_obs dict) and common methods (close, render, reset,
logging_sanity_check).

Subclasses must implement: reset_idx, step, compute_rewards_and_crashes, process_obs_for_task.
"""

from __future__ import annotations

from typing import Callable

from aerial_gym.task.base_task import BaseTask, StepReturn
from aerial_gym.task.task_config_protocol import TaskConfig
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.env_manager.env_manager import EnvManager
import torch
import numpy as np

from aerial_gym.utils.math import *  # noqa: F401,F403
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.vae.vae_image_encoder import VAEImageEncoder

from gym.spaces import Dict, Box

logger = CustomLogger("base_navigation_task")


class BaseNavigationTask(BaseTask):
    """Shared foundation for all navigation task variants."""

    sim_env: EnvManager
    vae_model: Callable[[torch.Tensor], torch.Tensor]
    task_obs: dict[str, torch.Tensor]
    obs_dict: dict[str, torch.Tensor]
    infos: dict[str, torch.Tensor]
    action_transformation_function: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        task_config: TaskConfig,
        seed: int | None = None,
        num_envs: int | None = None,
        headless: bool | None = None,
        device: str | None = None,
        use_warp: bool | None = None,
    ) -> None:
        # Apply caller overrides to config before passing to BaseTask
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = self.task_config.device

        # Convert reward parameters to tensors
        for key in self.task_config.reward_parameters:
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )

        # Build the simulation environment
        logger.info(
            "Building environment: sim=%s, env=%s, robot=%s, controller=%s",
            self.task_config.sim_name,
            self.task_config.env_name,
            self.task_config.robot_name,
            self.task_config.controller_name,
        )
        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )
        self.num_envs = self.sim_env.num_envs

        # Target position (subclass decides how to populate on reset)
        self.target_position = torch.zeros(
            (self.num_envs, 3), device=self.device, requires_grad=False
        )

        # Aggregate counters for curriculum evaluation
        self.success_aggregate = 0
        self.crashes_aggregate = 0
        self.timeouts_aggregate = 0

        # Position error tensors (vehicle frame)
        self.pos_error_vehicle_frame = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.pos_error_vehicle_frame_prev = torch.zeros(
            self.num_envs, 3, device=self.device
        )

        # VAE model (shared or identity)
        self._init_vae()

        # Retrieve initial observations from simulation
        self.obs_dict = self.sim_env.get_obs()

        # Curriculum level
        if "curriculum_level" not in self.obs_dict:
            self.curriculum_level = self.task_config.curriculum.min_level
            self.obs_dict["curriculum_level"] = self.curriculum_level
        else:
            self.curriculum_level = self.obs_dict["curriculum_level"]
        self.curriculum_progress_fraction = (
            self.curriculum_level - self.task_config.curriculum.min_level
        ) / max(
            1,
            self.task_config.curriculum.max_level
            - self.task_config.curriculum.min_level,
        )

        # Termination / truncation / reward tensors
        self.terminations = self.obs_dict.get(
            "terminations", self.obs_dict["crashes"]
        )
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.num_envs, device=self.device)

        # Observation and action spaces
        self._init_spaces()

        # Task observation buffers
        self._init_task_obs()

        # Step counter
        self.num_task_steps = 0

        # Info dict (populated each step)
        self.infos = {}

    # ------------------------------------------------------------------
    # Private init helpers
    # ------------------------------------------------------------------

    def _init_vae(self) -> None:
        """Set up VAE encoder (or identity fallback) and latent buffers."""
        if self.task_config.vae_config.use_vae:
            self.vae_model = VAEImageEncoder(
                config=self.task_config.vae_config, device=self.device
            )
            latent_dims = self.task_config.vae_config.latent_dims
        else:
            self.vae_model = lambda x: x
            latent_dims = 1

        self.image_latents = torch.zeros(
            (self.num_envs, latent_dims),
            device=self.device,
            requires_grad=False,
        )

    def _init_spaces(self) -> None:
        """Define Gymnasium observation and action spaces."""
        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                ),
                "image_obs": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(1, 270, 480),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_transformation_function = (
            self.task_config.action_transformation_function
        )

    def _init_task_obs(self) -> None:
        """Allocate task observation tensor buffers."""
        self.task_obs = {
            "observations": torch.zeros(
                (self.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (self.num_envs, self.task_config.privileged_observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

    # ------------------------------------------------------------------
    # Common methods
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.sim_env.delete_env()

    def render(self) -> bool:
        return self.sim_env.render()

    def reset(self) -> StepReturn:
        self.reset_idx(torch.arange(self.num_envs))
        return self.get_return_tuple()

    def get_return_tuple(self) -> StepReturn:
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def logging_sanity_check(self, infos: dict[str, torch.Tensor]) -> None:
        """Detect impossible episode outcome combinations."""
        successes = infos["successes"]
        crashes = infos["crashes"]
        timeouts = infos["timeouts"]
        time_at_crash = torch.where(
            crashes > 0,
            self.sim_env.sim_steps,
            self.task_config.episode_len_steps * torch.ones_like(self.sim_env.sim_steps),
        )
        env_list_for_toc = (time_at_crash < 5).nonzero(as_tuple=False).squeeze(-1)
        if len(env_list_for_toc) > 0:
            logger.critical("Crash is happening too soon.")
            logger.critical(f"Envs crashing too soon: {env_list_for_toc}")
            logger.critical(f"Time at crash: {time_at_crash[env_list_for_toc]}")

        if torch.sum(torch.logical_and(successes, crashes)) > 0:
            logger.critical("Success and crash are occurring at the same time")
        if torch.sum(torch.logical_and(successes, timeouts)) > 0:
            logger.critical("Success and timeout are occurring at the same time")
        if torch.sum(torch.logical_and(crashes, timeouts)) > 0:
            logger.critical("Crash and timeout are occurring at the same time")

    def process_image_observation(self) -> None:
        """Encode depth image through VAE (or pass through)."""
        image_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
        if self.task_config.vae_config.use_vae:
            self.image_latents[:] = self.vae_model.encode(image_obs)
