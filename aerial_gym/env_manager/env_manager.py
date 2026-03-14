from __future__ import annotations


from aerial_gym.env_manager.IGE_env_manager import IsaacGymEnv

from aerial_gym.env_manager.base_env_manager import BaseManager
from aerial_gym.env_manager.asset_manager import AssetManager
from aerial_gym.env_manager.warp_env_manager import WarpEnv
from aerial_gym.env_manager.asset_loader import AssetLoader
from aerial_gym.robots.robot_manager import RobotManagerIGE
from aerial_gym.env_manager.obstacle_manager import ObstacleManager


from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.sim_registry import sim_config_registry
from aerial_gym.registry.robot_registry import robot_registry

import torch

from aerial_gym.utils.logging import CustomLogger

import math, random

logger = CustomLogger("env_manager")


class EnvManager(BaseManager):
    """
    This class manages the environment. This can handle the creation of the
    robot, environment, and asset managers. This class handles the tensor creation and destruction.

    Moreover, the environment manager can be called within the main environment
    class to manipulate the environment by abstracting the interface.

    This script can remain as generic as possible to handle different types of
    environments, while changes can be made in the individual robot or environment
    managers to handle specific cases.
    """

    def __init__(
        self,
        sim_name: str,
        env_name: str,
        robot_name: str,
        controller_name: str,
        device: str,
        args: object = None,
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

        self.global_tensor_dict = {}

        logger.info("Populating environments.")
        self.populate_env(env_cfg=self.cfg, sim_cfg=self.sim_config)
        logger.info("[DONE] Populating environments.")
        self.prepare_sim()

        self.sim_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, requires_grad=False, device=self.device
        )

    def create_sim(self, env_cfg: object, sim_cfg: object) -> None:
        """
        This function creates the environment and the robot manager. Does the necessary things to create the environment
        for an IsaacGym environment instance.
        """
        logger.info("Creating simulation instance.")
        logger.info("Instantiating IGE object.")

        # === Need to check this here otherwise IGE will crash with segfault for different CUDA GPUs ====
        has_IGE_cameras = False
        robot_config = robot_registry.get_robot_config(self.robot_name)
        if robot_config.sensor_config.enable_camera == True and self.use_warp == False:
            has_IGE_cameras = True
        # ===============================================================================================

        self.IGE_env = IsaacGymEnv(env_cfg, sim_cfg, has_IGE_cameras, self.device)

        # define a global dictionary to store the simulation objects and important parameters
        # that are shared across the environment, asset, and robot managers
        self.global_sim_dict = {}
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

    def populate_env(self, env_cfg: object, sim_cfg: object) -> None:
        """
        This function populates the environment with the necessary assets and robots.
        """
        # create the simulation instance with the environment and robot manager
        self.create_sim(env_cfg, sim_cfg)

        self.robot_manager.create_robot(self.asset_loader)

        # first select assets for the environments:
        self.global_asset_dicts, keep_in_env_num = self.asset_loader.select_assets_for_sim()

        if self.keep_in_env == None:
            self.keep_in_env = keep_in_env_num
        elif self.keep_in_env != keep_in_env_num:
            raise Exception(
                "Inconsistent number of assets kept in the environment. The number of keep_in_env assets must be equal for all environments. Check."
            )

        # add the assets to the environment
        segmentation_ctr = 100

        self.global_asset_counter = 0
        self.step_counter = 0

        self.asset_min_state_ratio = None
        self.asset_max_state_ratio = None

        self.global_tensor_dict["crashes"] = torch.zeros(
            (self.num_envs), device=self.device, requires_grad=False, dtype=torch.bool
        )
        # Dedicated terminations tensor (true MDP terminals: success, boundary violation, etc.)
        self.global_tensor_dict["terminations"] = torch.zeros(
            (self.num_envs), device=self.device, requires_grad=False, dtype=torch.bool
        )
        self.global_tensor_dict["truncations"] = torch.zeros(
            (self.num_envs), device=self.device, requires_grad=False, dtype=torch.bool
        )

        self.num_env_actions = self.cfg.env.num_env_actions
        self.global_tensor_dict["num_env_actions"] = self.num_env_actions
        self.global_tensor_dict["env_actions"] = None
        self.global_tensor_dict["prev_env_actions"] = None

        self.collision_tensor = self.global_tensor_dict["crashes"]
        self.termination_tensor = self.global_tensor_dict["terminations"]
        self.truncation_tensor = self.global_tensor_dict["truncations"]

        # Initialize per-env gate variant selection counters for deterministic randomization
        try:
            self.global_tensor_dict["gate_variant_counter"] = torch.zeros(
                (self.num_envs), device=self.device, dtype=torch.int64
            )
        except Exception:
            pass

        # Before we populate the environment, we need to create the ground plane
        if self.cfg.env.create_ground_plane:
            logger.info("Creating ground plane in Isaac Gym Simulation.")
            self.IGE_env.create_ground_plane()
            logger.info("[DONE] Creating ground plane in Isaac Gym Simulation")

        # Track per-env list of gate variant asset indices after creation
        self.global_tensor_dict["gate_variant_indices_per_env"] = [[] for _ in range(self.cfg.env.num_envs)]
        self.global_tensor_dict["active_gate_variant_index"] = torch.full((self.cfg.env.num_envs,), -1, device=self.device, dtype=torch.long)
        self.global_tensor_dict["active_gate_variant_array_index"] = torch.full((self.cfg.env.num_envs,), -1, device=self.device, dtype=torch.long)
        self.global_tensor_dict["gate_variant_names_per_env"] = [[] for _ in range(self.cfg.env.num_envs)]

        for i in range(self.cfg.env.num_envs):
            logger.debug(f"Populating environment {i}")
            if i % 1000 == 0:
                logger.info(f"Populating environment {i}")

            env_handle = self.IGE_env.create_env(i)
            if self.cfg.env.use_warp:
                self.warp_env.create_env(i)

            # add robot asset in the environment
            self.robot_manager.add_robot_to_env(
                self.IGE_env, env_handle, self.global_asset_counter, i, segmentation_ctr
            )
            self.global_asset_counter += 1

            self.num_obs_in_env = 0
            # add regular assets in the environment
            local_gate_variant_indices = []  # LOCAL indices including robot (robot=0)
            local_gate_variant_names = []
            local_index_counter = 1  # robot is 0, first env asset starts at 1
            for asset_info_dict in self.global_asset_dicts[i]:
                asset_handle, ige_seg_ctr = self.IGE_env.add_asset_to_env(
                    asset_info_dict,
                    env_handle,
                    i,
                    self.global_asset_counter,
                    segmentation_ctr,
                )
                # Track gate variants using LOCAL per-env index (including robot)
                if asset_info_dict.get("is_gate_variant", False):
                    local_gate_variant_indices.append(local_index_counter)
                    gate_variant_name = asset_info_dict.get("gate_variant_name", "unknown")
                    local_gate_variant_names.append(gate_variant_name)
                    # logger.warning(f"[GATE_DEBUG] Env {i}: Found gate variant '{gate_variant_name}' at local index {local_index_counter} (file: {asset_info_dict.get('filename', 'unknown')})")
                self.num_obs_in_env += 1
                warp_segmentation_ctr = 0
                if self.cfg.env.use_warp:
                    empty_handle, warp_segmentation_ctr = self.warp_env.add_asset_to_env(
                        asset_info_dict,
                        i,
                        self.global_asset_counter,
                        segmentation_ctr,
                    )
                # Update counters after adding this asset
                self.global_asset_counter += 1
                local_index_counter += 1
                segmentation_ctr += max(ige_seg_ctr, warp_segmentation_ctr)
                if self.asset_min_state_ratio is None or self.asset_max_state_ratio is None:
                    self.asset_min_state_ratio = torch.tensor(
                        asset_info_dict["min_state_ratio"], requires_grad=False
                    ).unsqueeze(0)
                    self.asset_max_state_ratio = torch.tensor(
                        asset_info_dict["max_state_ratio"], requires_grad=False
                    ).unsqueeze(0)
                else:
                    self.asset_min_state_ratio = torch.vstack(
                        (
                            self.asset_min_state_ratio,
                            torch.tensor(asset_info_dict["min_state_ratio"], requires_grad=False),
                        )
                    )
                    self.asset_max_state_ratio = torch.vstack(
                        (
                            self.asset_max_state_ratio,
                            torch.tensor(asset_info_dict["max_state_ratio"], requires_grad=False),
                        )
                    )
            # Persist gate variant indices for this env
            self.global_tensor_dict["gate_variant_indices_per_env"][i] = local_gate_variant_indices
            self.global_tensor_dict["gate_variant_names_per_env"][i] = local_gate_variant_names
            # logger.warning(f"[GateVariant] Env {i}: discovered {len(local_gate_variant_indices)} gate variants -> {local_gate_variant_names}")
            
            # DEBUG: Count different asset types for this environment
            # gate_count = len([name for name in local_gate_variant_names if "gate_scale_" in name])
            # wall_count = len([asset_dict for asset_dict in self.global_asset_dicts[i] if "wall" in asset_dict.get("asset_type", "")])
            # object_count = len([asset_dict for asset_dict in self.global_asset_dicts[i] if asset_dict.get("asset_type", "") == "objects"])
            # other_count = len(self.global_asset_dicts[i]) - gate_count - wall_count - object_count
            
            # logger.warning(f"[OBSTACLE_DEBUG] Env {i} asset breakdown: {gate_count} gates, {wall_count} walls, {object_count} objects, {other_count} other, TOTAL: {len(self.global_asset_dicts[i])}")

        # check if environment has 0 objects. If so, skip this step
        if self.asset_min_state_ratio is not None:
            self.asset_min_state_ratio = self.asset_min_state_ratio.to(self.device)
            self.asset_max_state_ratio = self.asset_max_state_ratio.to(self.device)
            self.global_tensor_dict["asset_min_state_ratio"] = self.asset_min_state_ratio.view(
                self.cfg.env.num_envs, -1, 13
            )
            self.global_tensor_dict["asset_max_state_ratio"] = self.asset_max_state_ratio.view(
                self.cfg.env.num_envs, -1, 13
            )
        else:
            self.global_tensor_dict["asset_min_state_ratio"] = torch.zeros(
                (self.cfg.env.num_envs, 0, 13), device=self.device
            )
            self.global_tensor_dict["asset_max_state_ratio"] = torch.zeros(
                (self.cfg.env.num_envs, 0, 13), device=self.device
            )

        self.global_tensor_dict["num_obstacles_in_env"] = self.num_obs_in_env
        # logger.warning(f"[OBSTACLE_DEBUG] EnvManager setting num_obstacles_in_env = {self.num_obs_in_env} (this is total loaded assets per env)")

        # Initial activation: select a gate for each env
        self.apply_gate_variant_selection(env_ids=torch.arange(self.cfg.env.num_envs, device=self.device))

    def prepare_sim(self) -> None:
        """
        This function prepares the simulation for the environment.
        """
        if not self.IGE_env.prepare_for_simulation(self, self.global_tensor_dict):
            raise Exception("Failed to prepare the simulation")
        if self.cfg.env.use_warp:
            if not self.warp_env.prepare_for_simulation(self.global_tensor_dict):
                raise Exception("Failed to prepare the simulation")

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
        # logger.debug(f"Resetting environments {env_ids}.")
        self.IGE_env.reset_idx(env_ids)
        # Determine how many assets to keep visible this reset
        obstacle_count = self.global_tensor_dict["num_obstacles_in_env"]
        # When obstacle randomization is disabled, first let AssetManager place everything,
        # then enforce the exact fixed count after gate selection.
        try:
            obs_dis = bool(self.global_tensor_dict.get('obstacles_randomization/disabled', False))
        except Exception:
            obs_dis = False
        if obs_dis:
            # Show all assets for now so none are hidden prematurely
            try:
                env_asset_state = self.global_tensor_dict["unfolded_env_asset_state_tensor"].view(self.cfg.env.num_envs, -1, 13)
                total_assets = env_asset_state.shape[1]
            except Exception:
                total_assets = self.asset_manager.env_asset_state_tensor.shape[1]
            obstacle_count = int(total_assets)
        # logger.warning(f"[OBSTACLE_DEBUG] EnvManager.reset_idx calling asset_manager.reset_idx with obstacle_count={obstacle_count}")
        self.asset_manager.reset_idx(env_ids, obstacle_count)
        # IMPORTANT: After asset manager randomization, activate one gate variant per env
        self.apply_gate_variant_selection(env_ids=env_ids)
        # Enforce fixed obstacle count after gate selection if obstacle randomization is disabled
        if obs_dis:
            try:
                fixed_count = int(self.global_tensor_dict.get('obstacles_randomization/fixed_count', 0))
            except Exception:
                fixed_count = 0
            try:
                env_asset_state = self.global_tensor_dict["unfolded_env_asset_state_tensor"].view(self.cfg.env.num_envs, -1, 13)
                gate_indices_all = self.global_tensor_dict.get("gate_variant_indices_per_env", [])
                total_assets = env_asset_state.shape[1]
                # Assets in [0 : keep_in_env) are fixed (walls etc.) and must stay visible
                fixed_indices = set(range(int(self.keep_in_env or 0)))
                # For each env, keep exactly `fixed_count` candidate obstacles (non-fixed, non-gate)
                loop_envs = env_ids.tolist() if hasattr(env_ids, 'tolist') else list(env_ids)
                for env_id in loop_envs:
                    gate_indices = set(gate_indices_all[env_id]) if gate_indices_all else set()
                    candidates = [i for i in range(total_assets) if (i not in fixed_indices) and (i not in gate_indices)]
                    # Keep the first N candidates; hide the rest
                    keep = set(candidates[: max(0, fixed_count)])
                    for i in candidates:
                        if i in keep:
                            continue
                        env_asset_state[env_id, i, 0:3] = torch.tensor([-1000.0, -1000.0, -1000.0], device=self.device)
                # Write back
                self.global_tensor_dict["unfolded_env_asset_state_tensor"][:] = env_asset_state.view(-1, 13)
            except Exception:
                pass
        if self.cfg.env.use_warp:
            self.warp_env.reset_idx(env_ids)
        self.robot_manager.reset_idx(env_ids)
        self.IGE_env.write_to_sim()
        self.sim_steps[env_ids] = 0

    def apply_gate_variant_selection(self, env_ids: torch.Tensor | None) -> None:
        """
        Select exactly one gate variant to be visible per environment, and hide the others by moving them far away.
        This must be called after AssetManager.reset_idx so that our visibility settings are not overwritten.
        """
        if env_ids is None:
            return
        # Guard: only apply after prepare_for_simulation sets num_assets_per_env
        if not hasattr(self.IGE_env, 'num_assets_per_env'):
            logger.debug("[GateVariant] Skipping selection (IGE_env not prepared yet)")
            return
        if isinstance(env_ids, torch.Tensor):
            ids = env_ids.tolist()
        else:
            ids = list(env_ids)
        env_asset_state = self.global_tensor_dict["unfolded_env_asset_state_tensor"].view(self.cfg.env.num_envs, -1, 13)
        for env_id in ids:
            gate_indices = self.global_tensor_dict["gate_variant_indices_per_env"][env_id]
            gate_names = self.global_tensor_dict["gate_variant_names_per_env"][env_id]
            if not gate_indices:
                continue
            
            # Check ablation flags to optionally disable gate size randomization
            disable_rand = bool(self.global_tensor_dict.get('gate_randomization/disabled', False))
            if disable_rand:
                # Fixed-scale mode
                fixed_scale = int(self.global_tensor_dict.get('gate_randomization/fixed_scale_percent', 100))
                # Parse available (index, scale)
                parsed = []
                for j, name in enumerate(gate_names):
                    scale = 100
                    if isinstance(name, str) and "gate_scale_" in name:
                        try:
                            scale = int(name.replace("gate_scale_", ""))
                        except Exception:
                            scale = 100
                    parsed.append((j, scale, name))
                if parsed:
                    # Find exact or nearest scale
                    exact = [j for (j, s, _) in parsed if s == fixed_scale]
                    if exact:
                        chosen_idx = exact[0]
                    else:
                        nearest = min(parsed, key=lambda t: abs(t[1] - fixed_scale))
                        chosen_idx = nearest[0]
                else:
                    chosen_idx = 0
                chosen_local_index = gate_indices[chosen_idx]
                active_name = gate_names[chosen_idx] if gate_names and chosen_idx < len(gate_names) else "unknown"
                # Place only the chosen variant at center; others hidden
                for j, local_index in enumerate(gate_indices):
                    if local_index < 0 or local_index >= env_asset_state.shape[1]:
                        continue
                    if j == chosen_idx:
                        env_asset_state[env_id, local_index, 0:3] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
                    else:
                        env_asset_state[env_id, local_index, 0:3] = torch.tensor([-1000.0, -1000.0, -1000.0], device=self.device)
                self.global_tensor_dict["active_gate_variant_index"][env_id] = chosen_local_index
                self.global_tensor_dict["active_gate_variant_array_index"][env_id] = chosen_idx
                continue

            # Curriculum-gated unlocking of gate variants (threshold-based):
            # At level 3: allow all scales >= 80%
            # At level 23: allow all scales >= 60% (enforced lower bound)
            cur_level = self.global_tensor_dict.get("curriculum_level", 3)
            try:
                cur_level = int(cur_level.item()) if hasattr(cur_level, 'item') else int(cur_level)
            except Exception:
                cur_level = 3
            
            # Unlock smaller gate scales beyond level 23 up to 50% at level 33 during eval-stretch
            # Training: linear threshold from 80 -> 60 over levels 3..23 (never below 60)
            # Eval-stretch (if enabled via global_tensor_dict): continue linearly 60 -> 40 over 23..stretch_end_level
            eval_stretch_enabled = bool(self.global_tensor_dict.get('eval_stretch_enabled', False))
            stretch_end_level = int(self.global_tensor_dict.get('eval_stretch_end_level', 23))
            if cur_level <= 3:
                min_allowed_scale = 80
            elif cur_level >= 23 and (not eval_stretch_enabled or stretch_end_level <= 23):
                min_allowed_scale = 60
            else:
                if cur_level < 23:
                    frac = (cur_level - 3) / (23 - 3)
                    raw = 80 - frac * (80 - 60)
                else:
                    upper = max(23, stretch_end_level)
                    span = max(1, upper - 23)
                    frac = min(1.0, (cur_level - 23) / float(span))
                    raw = 60 - frac * (60 - 50)
                # Quantize down to nearest 2% step (80, 78, ..., 50)
                min_allowed_scale = int((int(raw) // 2) * 2)
                if min_allowed_scale < 50:
                    min_allowed_scale = 50
                if min_allowed_scale > 100:
                    min_allowed_scale = 100
            
            # Parse (index, scale, name) and keep those meeting the threshold
            parsed = []
            for j, name in enumerate(gate_names):
                scale = 100
                if isinstance(name, str) and "gate_scale_" in name:
                    try:
                        scale = int(name.replace("gate_scale_", ""))
                    except Exception:
                        scale = 100
                parsed.append((j, scale, name))
            # Allowed indices are those with scale >= min_allowed_scale
            allowed_js = [j for (j, scale, _) in parsed if scale >= min_allowed_scale]
            # If nothing matched (e.g., unusual naming), fall back to largest scales
            if not allowed_js:
                parsed.sort(key=lambda x: x[1], reverse=True)
                allowed_js = [j for (j, _, __) in parsed[:1]] if parsed else []
            
            # Deterministic randomization across runs: use torch RNG seeded globally and per-env counter
            if allowed_js:
                allowed_pairs = [(j, scale) for (j, scale, _) in parsed if j in allowed_js]
                unique_scales = sorted({scale for (_, scale) in allowed_pairs}, reverse=True)
                # Increment per-env counter to advance RNG in a deterministic manner
                try:
                    self.global_tensor_dict["gate_variant_counter"][env_id] += 1
                except Exception:
                    pass
                # Sample an index deterministically using torch's seeded RNG
                # Pick a scale bucket first
                scale_idx = int(torch.randint(low=0, high=len(unique_scales), size=(1,), device=self.device).item())
                chosen_scale = unique_scales[scale_idx]
                candidates = [j for (j, scale) in allowed_pairs if scale == chosen_scale]
                cand_idx = int(torch.randint(low=0, high=len(candidates), size=(1,), device=self.device).item())
                chosen_idx = candidates[cand_idx]
            else:
                chosen_idx = 0
            chosen_local_index = gate_indices[chosen_idx]
            active_name = gate_names[chosen_idx] if gate_names and chosen_idx < len(gate_names) else "unknown"
            
            # Extract scale from gate name for debugging
            scale_info = "unknown"
            if "gate_scale_" in active_name:
                scale_info = active_name.replace("gate_scale_", "") + "%"
            
            # logger.warning(f"[GATE_RANDOM_SELECT] Env {env_id}: RANDOMLY SELECTED gate variant '{active_name}' (scale: {scale_info}, local index {chosen_local_index}) from {len(allowed_js)} allowed / {len(gate_indices)} total")
            
            self.global_tensor_dict["active_gate_variant_index"][env_id] = chosen_local_index  # Asset index
            self.global_tensor_dict["active_gate_variant_array_index"][env_id] = chosen_idx  # Index in gate variant arrays
            # Iterate all gate variants and place only the chosen at center; others moved far away
            for j, local_index in enumerate(gate_indices):
                if local_index < 0 or local_index >= env_asset_state.shape[1]:
                    logger.debug(f"[GateVariant] Env {env_id}: local_index {local_index} out of bounds for assets {env_asset_state.shape[1]}")
                    continue
                gate_name = gate_names[j] if gate_names and j < len(gate_names) else f"gate_{j}"
                if j == chosen_idx:
                    # place at center (already configured by min/max), but ensure visible by mirroring center pose
                    env_asset_state[env_id, local_index, 0:3] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
                    # logger.warning(f"[GATE_VISIBILITY] Env {env_id}: SHOWING gate '{gate_name}' at center (0,0,0)")
                else:
                    # move out of bounds to hide
                    env_asset_state[env_id, local_index, 0:3] = torch.tensor([-1000.0, -1000.0, -1000.0], device=self.device)
                    # logger.warning(f"[GATE_VISIBILITY] Env {env_id}: HIDING gate '{gate_name}' at (-1000,-1000,-1000)")
        # Write back the unfolded state tensor
        self.global_tensor_dict["unfolded_env_asset_state_tensor"][:] = env_asset_state.view(-1, 13)
        # logger.warning("[GateVariant] Applied gate variant selection for env_ids: {}".format(ids))

    def log_memory_use(self) -> None:
        """
        This function logs the memory usage of the GPU.
        """
        logger.warning(
            f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0)/1024/1024/1024}GB"
        )
        logger.warning(
            f"torch.cuda.memory_reserved: {torch.cuda.memory_reserved(0)/1024/1024/1024}GB"
        )
        logger.warning(
            f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(0)/1024/1024/1024}GB"
        )

        # Calculate and system RAM usage used by the objects of this class
        total_memory = 0
        for key, value in self.__dict__.items():
            total_memory += value.__sizeof__()
        logger.warning(
            f"Total memory used by the objects of this class: {total_memory/1024/1024}MB"
        )

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
        # if warp is used, the warp environment applies the actions here
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
        """
        This function steps the simulation for the environment.
        actions: The actions that are sent to the robot.
        env_actions: The actions that are sent to the environment entities.
        """
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

    def get_obs(self) -> dict[str, object]:
        # Just return the dict of all tensors. Whatever the task needs can be used to compute the rewards.
        return self.global_tensor_dict
