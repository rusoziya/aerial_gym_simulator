from __future__ import annotations

import torch

from aerial_gym.control.controllers.base_lee_controller import *
from aerial_gym.utils.math import *


class FullyActuatedController(BaseLeeController):
    def __init__(self, config: object, num_envs: int, device: str) -> None:
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict: dict[str, torch.Tensor] | None = None) -> None:
        super().init_tensors(global_tensor_dict)

    def update(self, command_actions: torch.Tensor) -> torch.Tensor:
        """
        Fully actuated controller. Input is in the form of desired position and orientation.
        command_actions = [p_x, p_y, p_z, qx, qy, qz, qw]
        Position setpoint is in the world frame
        Orientation reference is w.r.t world frame
        """
        self.reset_commands()
        command_actions[:, 3:7] = normalize(command_actions[:, 3:7])
        self.accel[:] = self.compute_acceleration(
            command_actions[:, 0:3], torch.zeros_like(command_actions[:, 0:3])
        )
        forces = self.mass * (self.accel - self.gravity)
        self.wrench_command[:, 0:3] = quat_rotate_inverse(self.robot_orientation, forces)
        self.desired_quat[:] = command_actions[:, 3:]
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.desired_quat, torch.zeros_like(command_actions[:, 0:3])
        )
        return self.wrench_command
