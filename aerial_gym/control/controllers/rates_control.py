from __future__ import annotations

import torch

from aerial_gym.control.controllers.base_lee_controller import BaseLeeController


class LeeRatesController(BaseLeeController):
    def __init__(self, config: object, num_envs: int, device: str) -> None:
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict: dict[str, torch.Tensor] | None = None) -> None:
        super().init_tensors(global_tensor_dict)

    def update(self, command_actions: torch.Tensor) -> torch.Tensor:
        """
        Lee attitude controller
        :param robot_state: tensor of shape (num_envs, 13) with state of the robot
        :param command_actions: tensor of shape (num_envs, 4) with desired thrust, roll, pitch and yaw_rate command in vehicle frame
        :return: m*g normalized thrust and interial normalized torques
        """
        self.reset_commands()
        # quaternion desired
        self.wrench_command[:, 2] = (command_actions[:, 0] - self.gravity) * self.mass
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.robot_orientation, command_actions[:, 1:4]
        )

        return self.wrench_command
