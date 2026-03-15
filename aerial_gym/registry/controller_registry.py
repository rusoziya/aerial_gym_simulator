from __future__ import annotations

from collections.abc import KeysView


class ControllerRegistry:
    """Registry mapping controller names to controller classes and configs."""

    def __init__(self) -> None:
        self.controller_classes: dict[str, type] = {}
        self.controller_configs: dict[str, type] = {}

    def register_controller(
        self, controller_name: str, controller_class: type, controller_config: type
    ) -> None:
        self.controller_classes[controller_name] = controller_class
        self.controller_configs[controller_name] = controller_config

    def get_controller_class(self, controller_name: str) -> type:
        return self.controller_classes[controller_name]

    def get_controller_names(self) -> KeysView[str]:
        return self.controller_classes.keys()

    def get_controller_config(self, controller_name: str) -> type:
        return self.controller_configs[controller_name]

    def make_controller(
        self, controller_name: str, num_envs: int, device: str, mode: str = "robot"
    ) -> tuple[object, type]:
        if controller_name not in self.controller_classes:
            raise ValueError(
                f"Controller {controller_name} not found in controller registry. Available controllers are {self.controller_classes.keys()}"
            )
        return (
            self.controller_classes[controller_name](
                self.controller_configs[controller_name],
                num_envs,
                device,
            ),
            self.controller_configs[controller_name],
        )


controller_registry = ControllerRegistry()
