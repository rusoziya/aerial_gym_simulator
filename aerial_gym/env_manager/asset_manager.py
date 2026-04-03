from __future__ import annotations

import torch

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import (
    quat_from_euler_xyz_tensor,
    torch_interpolate_ratio,
    torch_rand_float_tensor,
)

logger = CustomLogger("asset_manager")
logger.setLevel("DEBUG")


class AssetManager:
    def __init__(self, global_tensor_dict: dict[str, torch.Tensor], num_keep_in_env: int) -> None:
        self.init_tensors(global_tensor_dict, num_keep_in_env)

    def init_tensors(
        self, global_tensor_dict: dict[str, torch.Tensor], num_keep_in_env: int
    ) -> None:
        self.env_asset_state_tensor = global_tensor_dict["env_asset_state_tensor"]
        self.asset_min_state_ratio = global_tensor_dict["asset_min_state_ratio"]
        self.asset_max_state_ratio = global_tensor_dict["asset_max_state_ratio"]
        self.env_bounds_min = (
            global_tensor_dict["env_bounds_min"]
            .unsqueeze(1)
            .expand(-1, self.env_asset_state_tensor.shape[1], -1)
        )
        self.env_bounds_max = (
            global_tensor_dict["env_bounds_max"]
            .unsqueeze(1)
            .expand(-1, self.env_asset_state_tensor.shape[1], -1)
        )
        self.num_keep_in_env = num_keep_in_env

    def prepare_for_sim(self) -> None:
        self.reset(self.num_keep_in_env)
        logger.warning(f"Number of obstacles to be kept in the environment: {self.num_keep_in_env}")

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        pass

    def post_physics_step(self) -> None:
        pass

    def step(self, actions: object) -> None:
        pass

    def reset(self, num_obstacles_per_env: int) -> None:
        self.reset_idx(torch.arange(self.env_asset_state_tensor.shape[0]), num_obstacles_per_env)

    def reset_idx(self, env_ids: torch.Tensor, num_obstacles_per_env: int = 0) -> None:
        if num_obstacles_per_env < self.num_keep_in_env:
            logger.info(
                "Number of obstacles required in the environment by the \
                  code is lesser than the minimum number of obstacles that the environment configuration specifies."
            )
            num_obstacles_per_env = self.num_keep_in_env

        sampled_asset_state_ratio = torch_rand_float_tensor(
            self.asset_min_state_ratio, self.asset_max_state_ratio
        )
        self.env_asset_state_tensor[env_ids, :, 0:3] = torch_interpolate_ratio(
            min=self.env_bounds_min,
            max=self.env_bounds_max,
            ratio=sampled_asset_state_ratio[..., 0:3],
        )[env_ids, :, 0:3]
        self.env_asset_state_tensor[env_ids, :, 3:7] = quat_from_euler_xyz_tensor(
            sampled_asset_state_ratio[env_ids, :, 3:6]
        )

        env_list: list[int] = env_ids.tolist()
        for eid in env_list:
            self.env_asset_state_tensor[eid, num_obstacles_per_env:, 0:3] = -1000.0
