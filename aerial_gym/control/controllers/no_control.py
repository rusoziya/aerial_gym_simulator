from __future__ import annotations

import torch


from aerial_gym.utils.logging import CustomLogger
from aerial_gym.control.control_allocation import ControlAllocator

logger = CustomLogger("no_control")


class NoControl:
    def __init__(self, config: object, num_envs: int, device: str) -> None:
        pass

    def init_tensors(self, global_tensor_dict: dict[str, torch.Tensor] | None = None) -> None:
        pass

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.update(*args, **kwargs)

    def reset_commands(self) -> None:
        pass

    def reset(self) -> None:
        self.reset_idx(env_ids=None)

    def reset_idx(self, env_ids: torch.Tensor | None) -> None:
        pass

    def randomize_params(self, env_ids: torch.Tensor) -> None:
        pass

    def update(self, command_actions: torch.Tensor) -> torch.Tensor:
        return command_actions
