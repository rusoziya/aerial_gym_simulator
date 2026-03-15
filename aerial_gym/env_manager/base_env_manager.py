from __future__ import annotations

from abc import ABC


class BaseManager(ABC):
    def __init__(self, config: object, device: str) -> None:
        self.cfg = config
        self.device: str = device

    def reset(self) -> None:
        raise NotImplementedError("reset not implemented")

    def reset_idx(self, env_ids: object) -> None:
        raise NotImplementedError("reset_idx not implemented")

    def pre_physics_step(self, actions: object) -> None:
        pass

    def step(self) -> None:
        raise NotImplementedError("step not implemented")

    def post_physics_step(self) -> None:
        pass

    def init_tensors(self, global_tensor_dict: dict[str, object]) -> None:
        pass
