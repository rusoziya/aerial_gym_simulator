from __future__ import annotations

from abc import ABC


class BaseManager(ABC):
    def __init__(self, config, device):
        self.cfg = config
        self.device = device

    def reset(self) -> None:
        raise NotImplementedError("reset not implemented")

    def reset_idx(self, env_ids) -> None:
        raise NotImplementedError("reset_idx not implemented")

    def pre_physics_step(self, actions) -> None:
        pass

    def step(self) -> None:
        raise NotImplementedError("step not implemented")

    def post_physics_step(self) -> None:
        pass

    def init_tensors(self, global_tensor_dict) -> None:
        pass
