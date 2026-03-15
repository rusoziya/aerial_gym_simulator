from __future__ import annotations

import argparse
import math
import random
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaacgym import gymapi

    from aerial_gym.env_manager.global_tensor_dict_schema import GlobalSimDict, GlobalTensorDict

from aerial_gym.env_manager.asset_loader import AssetLoader
from aerial_gym.env_manager.asset_manager import AssetManager
from aerial_gym.env_manager.base_env_manager import BaseManager
from aerial_gym.env_manager.gate_variant_selection import (
    apply_gate_variant_selection as _apply_gate_variant_selection,
)
from aerial_gym.env_manager.IGE_env_manager import IsaacGymEnv
from aerial_gym.env_manager.obstacle_manager import ObstacleManager
from aerial_gym.env_manager.tensor_state import TensorState
from aerial_gym.env_manager.warp_env_manager import WarpEnv
from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.registry.sim_registry import sim_config_registry
from aerial_gym.robots.robot_manager import RobotManagerIGE
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("env_manager")


class EnvManager(BaseManager):
    def __init__(
        self,
        sim_name: str,
        env_name: str,
        robot_name: str,
        controller_name: str,
        device: str,
        args: argparse.Namespace | None = None,
        num_envs: int | None = None,
        use_warp: bool | None = None,
        headless: bool | None = None,
    ) -> None:
        self.robot_name = robot_name
        self.controller_name = controller_name
        self.sim_config = sim_config_registry.make_sim(sim_name)

        super().__init__(env_config_registry.make_env(env_name), device)

        if num_envs is not None:
            self.cfg.env.num_envs = num_envs
        if use_warp is not None:
            self.cfg.env.use_warp = use_warp
        if headless is not None:
            self.sim_config.viewer.headless = headless

        self.num_envs = self.cfg.env.num_envs
        self.use_warp = self.cfg.env.use_warp

        self.asset_manager = None
        self.tensor_manager = None
        self.env_args = args

        self.keep_in_env = None

        self.global_tensor_dict: TensorState = TensorState()

        logger.info("Populating environments.")
        self.populate_env(env_cfg=self.cfg, sim_cfg=self.sim_config)
        logger.info("[DONE] Populating environments.")
        self.prepare_sim()

        self.sim_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, requires_grad=False, device=self.device
        )

    def create_sim(self, env_cfg: type, sim_cfg: type) -> None:
        logger.info("Creating simulation instance.")
        logger.info("Instantiating IGE object.")

        has_IGE_cameras = False
        robot_config = robot_registry.get_robot_config(self.robot_name)
        if robot_config.sensor_config.enable_camera and not self.use_warp:
            has_IGE_cameras = True

        self.IGE_env = IsaacGymEnv(env_cfg, sim_cfg, has_IGE_cameras, self.device)

        # define a global dictionary to store the simulation objects and important parameters
        # that are shared across the environment, asset, and robot managers
        self.global_sim_dict: GlobalSimDict = {}
        self.global_sim_dict["gym"] = self.IGE_env.gym
        self.global_sim_dict["sim"] = self.IGE_env.sim
        self.global_sim_dict["env_cfg"] = self.cfg
        self.global_sim_dict["use_warp"] = self.IGE_env.cfg.env.use_warp
        self.global_sim_dict["num_envs"] = self.IGE_env.cfg.env.num_envs
        self.global_sim_dict["sim_cfg"] = sim_cfg

        logger.info("IGE object instantiated.")

        if self.cfg.env.use_warp:
            logger.info("Creating warp environment.")
            self.warp_env = WarpEnv(self.global_sim_dict, self.device)
            logger.info("Warp environment created.")

        self.asset_loader = AssetLoader(self.global_sim_dict, self.device)

        logger.info("Creating robot manager.")
        self.robot_manager = RobotManagerIGE(
            self.global_sim_dict, self.robot_name, self.controller_name, self.device
        )
        self.global_sim_dict["robot_config"] = self.robot_manager.cfg
        logger.info("[DONE] Creating robot manager.")

        logger.info("[DONE] Creating simulation instance.")

    def populate_env(self, env_cfg: type, sim_cfg: type) -> None:
        self.create_sim(env_cfg, sim_cfg)
        self.robot_manager.create_robot(self.asset_loader)

        self.global_asset_dicts, keep_in_env_num = self.asset_loader.select_assets_for_sim()
        num_unique = len(self.asset_loader.asset_buffer)
        num_per_env = len(self.global_asset_dicts[0]) if self.global_asset_dicts else 0
        logger.info(
            f"Loaded {num_unique} unique assets, {num_per_env} per env, "
            f"{keep_in_env_num} kept permanently"
        )
        if self.keep_in_env is None:
            self.keep_in_env = keep_in_env_num
        elif self.keep_in_env != keep_in_env_num:
            raise ValueError(
                "Inconsistent number of assets kept in the environment. "
                "The number of keep_in_env assets must be equal for all environments."
            )

        self._init_populate_tensors()
        self._create_ground_plane_if_needed()
        self._init_gate_variant_tracking()

        segmentation_ctr = self._populate_all_environments()
        self._finalize_asset_state_ratios()

        self.global_tensor_dict["num_obstacles_in_env"] = self.num_obs_in_env
        self.apply_gate_variant_selection(
            env_ids=torch.arange(self.cfg.env.num_envs, device=self.device)
        )

    def _init_populate_tensors(self) -> None:
        """Initialize counters, state ratio accumulators, and global tensor dict entries."""
        self.global_asset_counter = 0
        self.step_counter = 0
        self.asset_min_state_ratio = None
        self.asset_max_state_ratio = None

        self.global_tensor_dict["crashes"] = torch.zeros(
            self.num_envs, device=self.device, requires_grad=False, dtype=torch.bool
        )
        self.global_tensor_dict["terminations"] = torch.zeros(
            self.num_envs, device=self.device, requires_grad=False, dtype=torch.bool
        )
        self.global_tensor_dict["truncations"] = torch.zeros(
            self.num_envs, device=self.device, requires_grad=False, dtype=torch.bool
        )

        self.num_env_actions = self.cfg.env.num_env_actions
        self.global_tensor_dict["num_env_actions"] = self.num_env_actions
        self.global_tensor_dict["env_actions"] = None
        self.global_tensor_dict["prev_env_actions"] = None

        self.collision_tensor = self.global_tensor_dict["crashes"]
        self.termination_tensor = self.global_tensor_dict["terminations"]
        self.truncation_tensor = self.global_tensor_dict["truncations"]

        self.global_tensor_dict["gate_variant_counter"] = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int64
        )

    def _create_ground_plane_if_needed(self) -> None:
        if self.cfg.env.create_ground_plane:
            logger.info("Creating ground plane in Isaac Gym Simulation.")
            self.IGE_env.create_ground_plane()
            logger.info("[DONE] Creating ground plane in Isaac Gym Simulation")

    def _init_gate_variant_tracking(self) -> None:
        n = self.cfg.env.num_envs
        self.global_tensor_dict["gate_variant_indices_per_env"] = [[] for _ in range(n)]
        self.global_tensor_dict["active_gate_variant_index"] = torch.full(
            (n,), -1, device=self.device, dtype=torch.long
        )
        self.global_tensor_dict["active_gate_variant_array_index"] = torch.full(
            (n,), -1, device=self.device, dtype=torch.long
        )
        self.global_tensor_dict["gate_variant_names_per_env"] = [[] for _ in range(n)]

    def _populate_all_environments(self) -> int:
        """Create environments and add robots + assets. Returns final segmentation counter."""
        segmentation_ctr = 100
        for i in range(self.cfg.env.num_envs):
            logger.debug(f"Populating environment {i}")
            if i % 1000 == 0:
                logger.info(f"Populating environment {i}")

            env_handle = self.IGE_env.create_env(i)
            if self.cfg.env.use_warp:
                self.warp_env.create_env(i)

            self.robot_manager.add_robot_to_env(
                self.IGE_env, env_handle, self.global_asset_counter, i, segmentation_ctr
            )
            self.global_asset_counter += 1

            segmentation_ctr = self._add_assets_to_env(i, env_handle, segmentation_ctr)
        return segmentation_ctr

    def _add_assets_to_env(
        self, env_idx: int, env_handle: gymapi.Env, segmentation_ctr: int
    ) -> int:
        """Add all assets for a single environment. Returns updated segmentation counter."""
        self.num_obs_in_env = 0
        local_gate_variant_indices: list[int] = []
        local_gate_variant_names: list[str] = []
        local_index_counter = 1

        for asset_info_dict in self.global_asset_dicts[env_idx]:
            asset_handle, ige_seg_ctr = self.IGE_env.add_asset_to_env(
                asset_info_dict, env_handle, env_idx, self.global_asset_counter, segmentation_ctr
            )
            if asset_info_dict.get("is_gate_variant", False):
                local_gate_variant_indices.append(local_index_counter)
                local_gate_variant_names.append(asset_info_dict.get("gate_variant_name", "unknown"))
            self.num_obs_in_env += 1

            warp_segmentation_ctr = 0
            if self.cfg.env.use_warp:
                _, warp_segmentation_ctr = self.warp_env.add_asset_to_env(
                    asset_info_dict, env_idx, self.global_asset_counter, segmentation_ctr
                )

            self.global_asset_counter += 1
            local_index_counter += 1
            segmentation_ctr += max(ige_seg_ctr, warp_segmentation_ctr)
            self._accumulate_state_ratios(asset_info_dict)

        self.global_tensor_dict["gate_variant_indices_per_env"][env_idx] = (
            local_gate_variant_indices
        )
        self.global_tensor_dict["gate_variant_names_per_env"][env_idx] = local_gate_variant_names
        return segmentation_ctr

    def _accumulate_state_ratios(
        self,
        asset_info_dict: dict[str, int | str | bool | list[float] | None],
    ) -> None:
        """Accumulate min/max state ratios from asset info."""
        ratio_tensor = torch.tensor(asset_info_dict["min_state_ratio"], requires_grad=False)
        max_tensor = torch.tensor(asset_info_dict["max_state_ratio"], requires_grad=False)
        if self.asset_min_state_ratio is None:
            self.asset_min_state_ratio = ratio_tensor.unsqueeze(0)
            self.asset_max_state_ratio = max_tensor.unsqueeze(0)
        else:
            self.asset_min_state_ratio = torch.vstack((self.asset_min_state_ratio, ratio_tensor))
            self.asset_max_state_ratio = torch.vstack((self.asset_max_state_ratio, max_tensor))

    def _finalize_asset_state_ratios(self) -> None:
        """Move state ratio tensors to device and store in global tensor dict."""
        n = self.cfg.env.num_envs
        if self.asset_min_state_ratio is not None:
            self.asset_min_state_ratio = self.asset_min_state_ratio.to(self.device)
            self.asset_max_state_ratio = self.asset_max_state_ratio.to(self.device)
            self.global_tensor_dict["asset_min_state_ratio"] = self.asset_min_state_ratio.view(
                n, -1, 13
            )
            self.global_tensor_dict["asset_max_state_ratio"] = self.asset_max_state_ratio.view(
                n, -1, 13
            )
        else:
            self.global_tensor_dict["asset_min_state_ratio"] = torch.zeros(
                (n, 0, 13), device=self.device
            )
            self.global_tensor_dict["asset_max_state_ratio"] = torch.zeros(
                (n, 0, 13), device=self.device
            )

    def prepare_sim(self) -> None:
        if not self.IGE_env.prepare_for_simulation(self, self.global_tensor_dict):
            raise RuntimeError("Failed to prepare the simulation")
        if self.cfg.env.use_warp:
            if not self.warp_env.prepare_for_simulation(self.global_tensor_dict):
                raise RuntimeError("Failed to prepare the simulation")

        self.asset_manager = AssetManager(self.global_tensor_dict, self.keep_in_env)
        self.asset_manager.prepare_for_sim()
        self.robot_manager.prepare_for_sim(self.global_tensor_dict)
        self.obstacle_manager = ObstacleManager(
            self.IGE_env.num_assets_per_env, self.cfg, self.device
        )
        self.obstacle_manager.prepare_for_sim(self.global_tensor_dict)
        self.num_robot_actions = self.global_tensor_dict["num_robot_actions"]

    def reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        """
        This function resets the environment for the given environment indices.
        """
        # first reset the Isaac Gym environment since that determines the environment bounds
        # then reset the asset managet that respositions assets within the environment
        # then reset the warp environment if it is being used that reads the state tensors from the assets and transforms meshes
        # finally reset the robot manager that resets the robot state tensors and the sensors
        self.IGE_env.reset_idx(env_ids)
        # Determine how many assets to keep visible this reset
        obstacle_count = self.global_tensor_dict["num_obstacles_in_env"]
        # When obstacle randomization is disabled, first let AssetManager place everything,
        # then enforce the exact fixed count after gate selection.
        try:
            obs_dis = bool(self.global_tensor_dict.get("obstacles_randomization/disabled", False))
        except (KeyError, TypeError):
            obs_dis = False
        if obs_dis:
            # Show all assets for now so none are hidden prematurely
            try:
                env_asset_state = self.global_tensor_dict["unfolded_env_asset_state_tensor"].view(
                    self.cfg.env.num_envs, -1, 13
                )
                total_assets = env_asset_state.shape[1]
            except (KeyError, TypeError):
                total_assets = self.asset_manager.env_asset_state_tensor.shape[1]
            obstacle_count = int(total_assets)
        self.asset_manager.reset_idx(env_ids, obstacle_count)
        # IMPORTANT: After asset manager randomization, activate one gate variant per env
        self.apply_gate_variant_selection(env_ids=env_ids)
        # Enforce fixed obstacle count after gate selection if obstacle randomization is disabled
        if obs_dis:
            self._enforce_fixed_obstacle_count(env_ids)
        if self.cfg.env.use_warp:
            self.warp_env.reset_idx(env_ids)
        self.robot_manager.reset_idx(env_ids)
        self.IGE_env.write_to_sim()
        self.sim_steps[env_ids] = 0

    def _enforce_fixed_obstacle_count(self, env_ids: torch.Tensor) -> None:
        """Hide excess obstacles so exactly `fixed_count` non-fixed, non-gate assets remain."""
        try:
            fixed_count = int(self.global_tensor_dict.get("obstacles_randomization/fixed_count", 0))
        except (ValueError, TypeError):
            fixed_count = 0
        env_asset_state = self.global_tensor_dict["unfolded_env_asset_state_tensor"].view(
            self.cfg.env.num_envs, -1, 13
        )
        gate_indices_all = self.global_tensor_dict.get("gate_variant_indices_per_env", [])
        total_assets = env_asset_state.shape[1]
        # Assets in [0 : keep_in_env) are fixed (walls etc.) and must stay visible
        fixed_indices = set(range(int(self.keep_in_env or 0)))
        # For each env, keep exactly `fixed_count` candidate obstacles (non-fixed, non-gate)
        loop_envs = env_ids.tolist()
        for env_id in loop_envs:
            gate_indices = set(gate_indices_all[env_id]) if gate_indices_all else set()
            candidates = [
                i
                for i in range(total_assets)
                if (i not in fixed_indices) and (i not in gate_indices)
            ]
            # Keep the first N candidates; hide the rest
            keep = set(candidates[: max(0, fixed_count)])
            for i in candidates:
                if i in keep:
                    continue
                env_asset_state[env_id, i, 0:3] = torch.tensor(
                    [-1000.0, -1000.0, -1000.0], device=self.device
                )
        # Write back
        self.global_tensor_dict["unfolded_env_asset_state_tensor"][:] = env_asset_state.view(-1, 13)

    def apply_gate_variant_selection(self, env_ids: torch.Tensor | None) -> None:
        """Select exactly one gate variant per environment, hiding the rest."""
        ige_prepared = self.IGE_env.num_assets_per_env > 0
        _apply_gate_variant_selection(
            env_ids=env_ids,
            global_tensor_dict=self.global_tensor_dict,
            num_envs=self.cfg.env.num_envs,
            device=self.device,
            ige_env_prepared=ige_prepared,
        )

    def log_memory_use(self) -> None:
        allocated_gb = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
        reserved_gb = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
        max_reserved_gb = torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024
        logger.warning(f"torch.cuda.memory_allocated: {allocated_gb}GB")
        logger.warning(f"torch.cuda.memory_reserved: {reserved_gb}GB")
        logger.warning(f"torch.cuda.max_memory_reserved: {max_reserved_gb}GB")
        total_memory = sum(v.__sizeof__() for v in self.__dict__.values())
        logger.warning(f"Total memory used by objects: {total_memory / 1024 / 1024}MB")

    def reset(self) -> None:
        self.reset_idx(env_ids=torch.arange(self.cfg.env.num_envs))

    def pre_physics_step(self, actions: torch.Tensor, env_actions: torch.Tensor | None) -> None:
        # first let the robot compute the actions
        self.robot_manager.pre_physics_step(actions)
        # then the asset manager applies the actions here
        self.asset_manager.pre_physics_step(env_actions)
        # apply actions to obstacle manager
        self.obstacle_manager.pre_physics_step(env_actions)
        # then the simulator applies them here
        self.IGE_env.pre_physics_step(actions)
        # If you change the mesh, refit() needs to be called (expensive).
        if self.use_warp:
            self.warp_env.pre_physics_step(actions)

    def reset_tensors(self) -> None:
        self.collision_tensor[:] = 0
        self.termination_tensor[:] = 0
        self.truncation_tensor[:] = 0

    def simulate(self, actions: torch.Tensor, env_actions: torch.Tensor | None) -> None:
        self.pre_physics_step(actions, env_actions)
        self.IGE_env.physics_step()
        self.post_physics_step(actions, env_actions)

    def post_physics_step(self, actions: torch.Tensor, env_actions: torch.Tensor | None) -> None:
        self.IGE_env.post_physics_step()
        self.robot_manager.post_physics_step()
        if self.use_warp:
            self.warp_env.post_physics_step()
        self.asset_manager.post_physics_step()

    def compute_observations(self) -> None:
        self.collision_tensor[:] += (
            torch.norm(self.global_tensor_dict["robot_contact_force_tensor"], dim=1)
            > self.cfg.env.collision_force_threshold
        )

    def reset_terminated_and_truncated_envs(self) -> torch.Tensor:
        collision_envs = self.collision_tensor.nonzero(as_tuple=False).squeeze(-1)
        termination_envs = self.termination_tensor.nonzero(as_tuple=False).squeeze(-1)
        truncation_envs = self.truncation_tensor.nonzero(as_tuple=False).squeeze(-1)
        # Reset on: collisions (if enabled), terminations (always), or truncations
        reset_mask = (
            (self.collision_tensor & bool(self.cfg.env.reset_on_collision))
            | self.termination_tensor
            | self.truncation_tensor
        )
        envs_to_reset = reset_mask.nonzero(as_tuple=False).squeeze(-1)
        # reset the environments that have a collision
        if len(envs_to_reset) > 0:
            self.reset_idx(envs_to_reset)
        return envs_to_reset

    def render(self, render_components: str = "sensors") -> None:
        if render_components == "viewer":
            self.render_viewer()
        elif render_components == "sensors":
            self.render_sensors()

    def render_sensors(self) -> None:
        # render sensors after the physics step
        if self.robot_manager.has_IGE_sensors:
            self.IGE_env.step_graphics()
        self.robot_manager.capture_sensors()

    def render_viewer(self) -> None:
        # render viewer GUI
        self.IGE_env.render_viewer()

    def post_reward_calculation_step(self) -> torch.Tensor:
        envs_to_reset = self.reset_terminated_and_truncated_envs()
        # render is performed after reset to ensure that the sensors are updated from the new robot state.
        self.render(render_components="sensors")
        return envs_to_reset

    def step(self, actions: torch.Tensor, env_actions: torch.Tensor | None = None) -> None:
        self.reset_tensors()
        if env_actions is not None:
            if self.global_tensor_dict["env_actions"] is None:
                self.global_tensor_dict["env_actions"] = env_actions
                self.global_tensor_dict["prev_env_actions"] = env_actions
                self.prev_env_actions = self.global_tensor_dict["prev_env_actions"]
                self.env_actions = self.global_tensor_dict["env_actions"]
            logger.warning(
                f"Env actions shape: {env_actions.shape}, Previous env actions shape: {self.env_actions.shape}"
            )
            self.prev_env_actions[:] = self.env_actions
            self.env_actions[:] = env_actions
        num_physics_step_per_env_step = max(
            math.floor(
                random.gauss(
                    self.cfg.env.num_physics_steps_per_env_step_mean,
                    self.cfg.env.num_physics_steps_per_env_step_std,
                )
            ),
            0,
        )
        for timestep in range(num_physics_step_per_env_step):
            self.simulate(actions, env_actions)
            self.compute_observations()
        self.sim_steps[:] = self.sim_steps[:] + 1
        # Expose sim steps to global tensor dict for modules needing a global time base (e.g., static camera yaw sweep)
        self.global_tensor_dict["sim_steps"] = self.sim_steps
        self.step_counter += 1
        if self.step_counter % self.cfg.env.render_viewer_every_n_steps == 0:
            self.render(render_components="viewer")

    def get_obs(
        self,
    ) -> GlobalTensorDict:
        # Just return the dict of all tensors. Whatever the task needs can be used to compute the rewards.
        return self.global_tensor_dict
