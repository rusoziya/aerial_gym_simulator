from __future__ import annotations

from collections.abc import KeysView


class SimConfigRegistry:
    """Registry mapping sim config names to config classes."""

    def __init__(self) -> None:
        self.sim_configs: dict[str, type] = {}

    def register(self, sim_name: str, sim_config: type) -> None:
        self.sim_configs[sim_name] = sim_config

    def get_sim_config(self, sim_name: str) -> type:
        return self.sim_configs[sim_name]

    def get_sim_names(self) -> KeysView[str]:
        return self.sim_configs.keys()

    def make_sim(self, sim_name: str) -> type:
        if sim_name not in self.sim_configs:
            raise ValueError(f"sim {sim_name} not found in sim registry")
        return self.sim_configs[sim_name]


sim_config_registry = SimConfigRegistry()
