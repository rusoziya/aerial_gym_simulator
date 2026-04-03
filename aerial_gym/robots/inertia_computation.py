from __future__ import annotations

import pytorch3d.transforms as p3d_transforms
import torch

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("inertia_computation")


def compute_robot_com(
    state_list: list[object],
    rbp: list[object],
    device: str,
) -> tuple[torch.Tensor, float]:
    """Compute the center of mass for the entire robot from rigid body properties."""
    robot_com = torch.zeros((1, 4), device=device)
    robot_mass = 0.0
    quat = torch.zeros((1, 4), dtype=torch.float32, device=device)
    transformation_mat = torch.zeros((4, 4), device=device)

    for item, properties in zip(state_list, rbp):
        obj_com = torch.zeros((1, 4), device=device)
        obj_mass = properties.mass
        obj_com[0, 0] = properties.com.x
        obj_com[0, 1] = properties.com.y
        obj_com[0, 2] = properties.com.z
        obj_com[0, 3] = 1.0

        position = item[0][0]
        rotation = item[0][1]

        quat[0, 0] = float(rotation[0])
        quat[0, 1] = float(rotation[1])
        quat[0, 2] = float(rotation[2])
        quat[0, 3] = float(rotation[3])

        rotmat = p3d_transforms.quaternion_to_matrix(quat[:, [3, 0, 1, 2]])[0]

        transformation_mat[0:3, 0:3] = rotmat
        transformation_mat[0, 3] = float(position[0])
        transformation_mat[1, 3] = float(position[1])
        transformation_mat[2, 3] = float(position[2])
        transformation_mat[3, 3] = 1.0

        obj_com_in_root_link_frame = torch.matmul(transformation_mat, obj_com.T).T
        logger.debug(f"Obj COM: {obj_com_in_root_link_frame}, Robot mass: {obj_mass}")

        robot_com += obj_mass * obj_com_in_root_link_frame
        robot_mass += obj_mass

    robot_com /= robot_mass
    robot_com[0, 3] = 1.0

    logger.debug(f"Robot COM: {robot_com}, Robot mass: {robot_mass}")
    return robot_com, robot_mass


def compute_composite_inertia(
    state_list: list[object],
    rbp: list[object],
    robot_com: torch.Tensor,
    device: str,
) -> tuple[float, torch.Tensor]:
    """Compute composite inertia using parallel axis theorem for all rigid bodies."""
    total_mass = 0.0
    total_inertia = torch.zeros((3, 3), device=device)

    quat = torch.zeros((1, 4), dtype=torch.float32, device=device)
    com = torch.zeros((1, 4), device=device)
    body_inertia = torch.zeros((3, 3), device=device)
    transformation_mat = torch.zeros((4, 4), device=device)

    for item_ctr, (item, properties) in enumerate(zip(state_list, rbp)):
        position = item[0][0]
        rotation = item[0][1]
        logger.debug(f"Item: {item_ctr} position: {position}, rotation: {rotation}")

        com[0, 0] = properties.com.x
        com[0, 1] = properties.com.y
        com[0, 2] = properties.com.z
        com[0, 3] = 1.0

        _fill_inertia_tensor(body_inertia, properties.inertia)

        quat[0, 0] = float(rotation[0])
        quat[0, 1] = float(rotation[1])
        quat[0, 2] = float(rotation[2])
        quat[0, 3] = float(rotation[3])

        rotmat = p3d_transforms.quaternion_to_matrix(quat[:, [3, 0, 1, 2]])[0]

        transformed_inertia = torch.matmul(rotmat, torch.matmul(body_inertia, rotmat.T))
        logger.debug(
            f"intial inertia: {body_inertia.view(1, 9)} \n transformed_inertia: {transformed_inertia.view(1, 9)}"
        )

        transformation_mat[0:3, 0:3] = rotmat
        transformation_mat[0, 3] = float(position[0])
        transformation_mat[1, 3] = float(position[1])
        transformation_mat[2, 3] = float(position[2])
        transformation_mat[3, 3] = 1.0

        com_in_root_link_frame = torch.matmul(transformation_mat, com.T).squeeze(1)
        obj_com_in_robot_com_frame = -(com_in_root_link_frame - robot_com.T.squeeze(1))
        obj_com_in_robot_com_frame[3] = 1.0

        logger.debug(f"COM in root link frame: {com_in_root_link_frame}")
        logger.debug(f"COM in robot COM frame: {obj_com_in_robot_com_frame}")

        _apply_parallel_axis_theorem(
            transformed_inertia, properties.mass, obj_com_in_robot_com_frame
        )

        total_mass += properties.mass
        total_inertia += transformed_inertia

    return total_mass, total_inertia


def _fill_inertia_tensor(body_inertia: torch.Tensor, inertia: object) -> None:
    """Fill a 3x3 inertia tensor from Isaac Gym inertia properties."""
    body_inertia[0, 0] = inertia.x.x
    body_inertia[0, 1] = inertia.x.y
    body_inertia[0, 2] = inertia.x.z
    body_inertia[1, 0] = inertia.y.x
    body_inertia[1, 1] = inertia.y.y
    body_inertia[1, 2] = inertia.y.z
    body_inertia[2, 0] = inertia.z.x
    body_inertia[2, 1] = inertia.z.y
    body_inertia[2, 2] = inertia.z.z


def _apply_parallel_axis_theorem(
    inertia: torch.Tensor,
    mass: float,
    offset: torch.Tensor,
) -> None:
    """Apply the parallel axis theorem to shift inertia to a new reference frame."""
    inertia[0, 0] += mass * (offset[1] ** 2 + offset[2] ** 2)
    inertia[1, 1] += mass * (offset[0] ** 2 + offset[2] ** 2)
    inertia[2, 2] += mass * (offset[0] ** 2 + offset[1] ** 2)
    inertia[0, 1] += -(mass * offset[0] * offset[1])
    inertia[0, 2] += -(mass * offset[0] * offset[2])
    inertia[1, 2] += -(mass * offset[1] * offset[2])
    inertia[1, 0] = inertia[0, 1]
    inertia[2, 0] = inertia[0, 2]
    inertia[2, 1] = inertia[1, 2]
