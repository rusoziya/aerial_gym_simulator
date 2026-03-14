from __future__ import annotations

from abc import ABC, abstractmethod

import time, torch, os, numpy as np

from aerial_gym.task.task_config_protocol import TaskConfig
from aerial_gym.utils.logging import CustomLogger
import random

logger = CustomLogger("base_task")

# Step return type used by all task subclasses
StepReturn = tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]


class BaseTask(ABC):
    def __init__(self, task_config: TaskConfig) -> None:
        self.task_config = task_config
        self.action_space = None
        self.observation_space = None
        self.reward_range = None
        self.metadata = None
        self.spec = None

        seed = task_config.seed
        if seed == -1:
            seed = time.time_ns() % (2**32)
        self.seed(seed)

    @abstractmethod
    def render(self, mode: str = "human") -> bool:
        raise NotImplementedError

    def seed(self, seed: int | None) -> None:
        if seed is None or seed < 0:
            logger.info(f"Seed is not valid. Will be sampled from system time.")
            seed = time.time_ns() % (2**32)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)

        logger.info("Setting seed: {}".format(seed))

    @abstractmethod
    def reset(self) -> StepReturn:
        raise NotImplementedError

    @abstractmethod
    def reset_idx(self, env_ids: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: torch.Tensor) -> StepReturn:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
