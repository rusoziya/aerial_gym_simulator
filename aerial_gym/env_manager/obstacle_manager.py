from __future__ import annotations

from aerial_gym.env_manager.base_env_manager import BaseManager

from aerial_gym.utils.math import *


from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("obstacle_manager")


class ObstacleManager(BaseManager):
    def __init__(self, num_assets, config, device):
        super().__init__(config, device)
        self.global_tensor_dict = {}
        self.num_assets = num_assets

        logger.debug("Obstacle Manager initialized")

    def prepare_for_sim(self, global_tensor_dict) -> None:
        if self.num_assets <= 1:
            return
        self.global_tensor_dict = global_tensor_dict
        self.obstacle_position = global_tensor_dict["obstacle_position"]
        self.obstacle_orientation = global_tensor_dict["obstacle_orientation"]
        self.obstacle_linvel = global_tensor_dict["obstacle_linvel"]
        self.obstacle_angvel = global_tensor_dict["obstacle_angvel"]

    def reset(self) -> None:
        return

    def reset_idx(self, env_ids) -> None:
        return

    def pre_physics_step(self, actions=None) -> None:
        if self.num_assets <= 1 or actions is None:
            return
        self.obstacle_linvel[:] = actions[:, :, 0:3]
        self.obstacle_angvel[:] = actions[:, :, 3:6]

    def step(self) -> None:
        pass
