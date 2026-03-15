from __future__ import annotations

from collections.abc import KeysView


class EnvConfigRegistry:
    """Registry mapping environment config names to config classes."""

    def __init__(self) -> None:
        self.env_configs: dict[str, type] = {}

    def register(self, env_name: str, env_config: type) -> None:
        self.env_configs[env_name] = env_config

    def get_env_config(self, env_name: str) -> type:
        return self.env_configs[env_name]

    def get_env_names(self) -> KeysView[str]:
        return self.env_configs.keys()

    def make_env(self, env_name: str) -> type:
        if env_name not in self.env_configs:
            raise ValueError(f"env {env_name} not found in env registry")
        return self.env_configs[env_name]


env_config_registry = EnvConfigRegistry()
