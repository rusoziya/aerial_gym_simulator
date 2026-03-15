from __future__ import annotations

import torch
from isaacgym import gymtorch

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("tensor_population")


def populate_robot_tensors(
    global_tensor_dict: dict[str, object],
) -> None:
    """Populate robot-specific tensor slices from the root state tensor."""
    global_tensor_dict["robot_position"] = global_tensor_dict["robot_state_tensor"][:, :3]
    global_tensor_dict["robot_orientation"] = global_tensor_dict["robot_state_tensor"][:, 3:7]
    global_tensor_dict["robot_linvel"] = global_tensor_dict["robot_state_tensor"][:, 7:10]
    global_tensor_dict["robot_angvel"] = global_tensor_dict["robot_state_tensor"][:, 10:]
    global_tensor_dict["robot_body_angvel"] = torch.zeros_like(
        global_tensor_dict["robot_state_tensor"][:, 10:13]
    )
    global_tensor_dict["robot_body_linvel"] = torch.zeros_like(
        global_tensor_dict["robot_state_tensor"][:, 7:10]
    )
    global_tensor_dict["robot_euler_angles"] = torch.zeros_like(
        global_tensor_dict["robot_state_tensor"][:, 7:10]
    )


def populate_obstacle_tensors(
    global_tensor_dict: dict[str, object],
    num_envs: int,
    num_rigid_bodies_per_env: int,
    num_rigid_bodies_robot: int,
    device: str,
) -> None:
    """Populate obstacle-specific tensor slices from the environment asset state tensor."""
    global_tensor_dict["obstacle_position"] = global_tensor_dict["env_asset_state_tensor"][
        :, :, 0:3
    ]
    global_tensor_dict["obstacle_orientation"] = global_tensor_dict["env_asset_state_tensor"][
        :, :, 3:7
    ]
    global_tensor_dict["obstacle_linvel"] = global_tensor_dict["env_asset_state_tensor"][
        :, :, 7:10
    ]
    global_tensor_dict["obstacle_angvel"] = global_tensor_dict["env_asset_state_tensor"][
        :, :, 10:
    ]
    global_tensor_dict["obstacle_body_angvel"] = torch.zeros_like(
        global_tensor_dict["env_asset_state_tensor"][:, :, 10:13]
    )
    global_tensor_dict["obstacle_body_linvel"] = torch.zeros_like(
        global_tensor_dict["env_asset_state_tensor"][:, :, 7:10]
    )
    global_tensor_dict["obstacle_euler_angles"] = torch.zeros_like(
        global_tensor_dict["env_asset_state_tensor"][:, :, 7:10]
    )

    idx = num_rigid_bodies_robot
    global_tensor_dict["obstacle_force_tensor"] = global_tensor_dict["global_force_tensor"].view(
        num_envs, num_rigid_bodies_per_env, 3
    )[:, idx:, :]
    global_tensor_dict["obstacle_torque_tensor"] = global_tensor_dict["global_torque_tensor"].view(
        num_envs, num_rigid_bodies_per_env, 3
    )[:, idx:, :]


def populate_force_tensors(
    global_tensor_dict: dict[str, object],
    gym: object,
    sim: object,
    num_envs: int,
    num_rigid_bodies_per_env: int,
    num_rigid_bodies_robot: int,
    device: str,
) -> None:
    """Populate rigid body state, force, torque, DOF, and contact tensors."""
    global_tensor_dict["rigid_body_state_tensor"] = gymtorch.wrap_tensor(
        gym.acquire_rigid_body_state_tensor(sim)
    )
    global_tensor_dict["global_force_tensor"] = torch.zeros(
        (global_tensor_dict["rigid_body_state_tensor"].shape[0], 3),
        device=device,
        requires_grad=False,
    )
    global_tensor_dict["global_torque_tensor"] = torch.zeros(
        (global_tensor_dict["rigid_body_state_tensor"].shape[0], 3),
        device=device,
        requires_grad=False,
    )

    idx = num_rigid_bodies_robot
    global_tensor_dict["robot_force_tensor"] = global_tensor_dict["global_force_tensor"].view(
        num_envs, num_rigid_bodies_per_env, 3
    )[:, :idx, :]
    global_tensor_dict["robot_torque_tensor"] = global_tensor_dict["global_torque_tensor"].view(
        num_envs, num_rigid_bodies_per_env, 3
    )[:, :idx, :]
