from __future__ import annotations

from collections.abc import KeysView

from aerial_gym.registry.controller_registry import controller_registry


class RobotRegistry:
    """Registry mapping robot names to robot classes and configs."""

    def __init__(self) -> None:
        self.robot_classes: dict[str, type] = {}
        self.robot_configs: dict[str, type] = {}

    def register(self, robot_name: str, robot_class: type, robot_config: type) -> None:
        self.robot_classes[robot_name] = robot_class
        self.robot_configs[robot_name] = robot_config

    def get_robot_class(self, robot_name: str) -> type:
        return self.robot_classes[robot_name]

    def get_robot_config(self, robot_name: str) -> type:
        return self.robot_configs[robot_name]

    def get_robot_names(self) -> KeysView[str]:
        return self.robot_classes.keys()

    def make_robot(self, robot_name: str, controller_name: str, env_config: object, device: str) -> tuple[object, type]:
        if robot_name not in self.robot_classes:
            raise ValueError(f"Robot {robot_name} not found in robot registry")
        return (
            self.robot_classes[robot_name](
                self.robot_configs[robot_name], controller_name, env_config, device
            ),
            self.robot_configs[robot_name],
        )


robot_registry = RobotRegistry()
