from __future__ import annotations

import torch


class CurriculumManager:
    def __init__(
        self,
        num_envs: int,
        min_level: int,
        max_level: int,
        level_step: int,
        device: str = "cuda:0",
    ) -> None:
        self.num_envs = num_envs
        self.min_level = min_level
        self.max_level = max_level
        self.level_step = level_step
        self.current_level = min_level
        self.device = device
        self.level_list = self._create_level_list()
        self.max_level_obtained = max(self.current_level, 0)

    def _create_level_list(self) -> list[int]:
        level_list = []
        for i in range(self.min_level, self.max_level + 1, self.level_step):
            level_list.append(i)
        return level_list

    def increase_curriculum_level(self) -> None:
        self.current_level = min(self.current_level + self.level_step, self.max_level)
        self.max_level_obtained = max(self.current_level, self.max_level_obtained)

    def get_current_level(self) -> int:
        return self.current_level

    def decrease_curriculum_level(self) -> None:
        self.current_level = max(self.current_level - self.level_step, self.min_level)
