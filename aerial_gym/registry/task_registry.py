from __future__ import annotations


class TaskRegistry:
    def __init__(self) -> None:
        self.task_class_registry: dict[str, type] = {}
        self.task_config_registry: dict[str, type] = {}

    def register_task(self, task_name: str, task_class: type, task_config: type) -> None:
        self.task_class_registry[task_name] = task_class
        self.task_config_registry[task_name] = task_config

    def get_task_class(self, task_name: str) -> type:
        return self.task_class_registry[task_name]

    def get_task_config(self, task_name: str) -> type:
        return self.task_config_registry[task_name]

    def get_task_names(self) -> list[str]:
        return list(self.task_class_registry.keys())

    def get_task_classes(self) -> list[type]:
        return list(self.task_class_registry.values())

    def get_task_configs(self) -> list[type]:
        return list(self.task_config_registry.values())

    def make_task(
        self,
        task_name: str,
        seed: int | None = None,
        num_envs: int | None = None,
        headless: bool | None = None,
        use_warp: bool | None = None,
    ) -> object:
        task_class = self.get_task_class(task_name)
        task_config = self.get_task_config(task_name)
        return task_class(
            task_config, seed=seed, num_envs=num_envs, headless=headless, use_warp=use_warp
        )


task_registry = TaskRegistry()
