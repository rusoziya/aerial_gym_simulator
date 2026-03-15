from __future__ import annotations

import torch

from aerial_gym.env_manager.env_manager import EnvManager


class SimBuilder:
    def __init__(self) -> None:
        self.sim_name: str | None = None
        self.env_name: str | None = None
        self.robot_name: str | None = None
        self.env: EnvManager | None = None

    def delete_env(self) -> None:
        del self.env
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.env = None

    def build_env(
        self,
        sim_name: str,
        env_name: str,
        robot_name: str,
        controller_name: str,
        device: str,
        args: object = None,
        num_envs: int | None = None,
        use_warp: bool | None = None,
        headless: bool | None = None,
    ) -> EnvManager:
        self.sim_name = sim_name
        self.env_name = env_name
        self.robot_name = robot_name
        self.env = EnvManager(
            sim_name=sim_name,
            env_name=env_name,
            robot_name=robot_name,
            controller_name=controller_name,
            args=args,
            device=device,
            num_envs=num_envs,
            use_warp=use_warp,
            headless=headless,
        )
        return self.env
