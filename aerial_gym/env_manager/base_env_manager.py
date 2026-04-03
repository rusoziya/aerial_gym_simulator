from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from aerial_gym.env_manager.global_tensor_dict_schema import GlobalTensorDict


class BaseManager(ABC):
    def __init__(self, config: type, device: str) -> None:
        self.cfg = config
        self.device: str = device

    def reset(self) -> None:
        raise NotImplementedError("reset not implemented")

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        raise NotImplementedError("reset_idx not implemented")

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        pass

    def step(self) -> None:
        raise NotImplementedError("step not implemented")

    def post_physics_step(self) -> None:
        pass

    def init_tensors(
        self,
        global_tensor_dict: GlobalTensorDict,
    ) -> None:
        pass
