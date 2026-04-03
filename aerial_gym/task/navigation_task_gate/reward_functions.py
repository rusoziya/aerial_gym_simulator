from __future__ import annotations

import torch

from aerial_gym.utils.math import exponential_penalty_function, exponential_reward_function


@torch.jit.script
def _compute_position_reward(
    dist: torch.Tensor,
    parameter_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    # type: (Tensor, Dict[str, Tensor]) -> Tensor
    return exponential_reward_function(
        parameter_dict["pos_reward_magnitude"],
        parameter_dict["pos_reward_exponent"],
        dist,
    )


@torch.jit.script
def _compute_very_close_reward(
    dist: torch.Tensor,
    parameter_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    # type: (Tensor, Dict[str, Tensor]) -> Tensor
    return exponential_reward_function(
        parameter_dict["very_close_to_goal_reward_magnitude"],
        parameter_dict["very_close_to_goal_reward_exponent"],
        dist,
    )


@torch.jit.script
def _compute_getting_closer_reward(
    dist: torch.Tensor,
    prev_dist_to_goal: torch.Tensor,
    parameter_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    # type: (Tensor, Tensor, Dict[str, Tensor]) -> Tensor
    getting_closer = prev_dist_to_goal - dist
    return torch.where(
        getting_closer > 0,
        parameter_dict["getting_closer_reward_multiplier"] * getting_closer,
        2.0 * parameter_dict["getting_closer_reward_multiplier"] * getting_closer,
    )


@torch.jit.script
def _compute_action_penalty(
    action: torch.Tensor,
    prev_action: torch.Tensor,
    parameter_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    # type: (Tensor, Tensor, Dict[str, Tensor]) -> Tensor
    action_diff = action - prev_action
    x_diff_penalty = exponential_penalty_function(
        parameter_dict["x_action_diff_penalty_magnitude"],
        parameter_dict["x_action_diff_penalty_exponent"],
        action_diff[:, 0],
    )
    y_diff_penalty = exponential_penalty_function(
        parameter_dict["y_action_diff_penalty_magnitude"],
        parameter_dict["y_action_diff_penalty_exponent"],
        action_diff[:, 1],
    )
    z_diff_penalty = exponential_penalty_function(
        parameter_dict["z_action_diff_penalty_magnitude"],
        parameter_dict["z_action_diff_penalty_exponent"],
        action_diff[:, 2],
    )
    yawrate_diff_penalty = exponential_penalty_function(
        parameter_dict["yawrate_action_diff_penalty_magnitude"],
        parameter_dict["yawrate_action_diff_penalty_exponent"],
        action_diff[:, 3],
    )
    action_diff_penalty = x_diff_penalty + y_diff_penalty + z_diff_penalty + yawrate_diff_penalty

    x_absolute_penalty = exponential_penalty_function(
        parameter_dict["x_absolute_action_penalty_magnitude"],
        parameter_dict["x_absolute_action_penalty_exponent"],
        action[:, 0],
    )
    y_absolute_penalty = exponential_penalty_function(
        parameter_dict["y_absolute_action_penalty_magnitude"],
        parameter_dict["y_absolute_action_penalty_exponent"],
        action[:, 1],
    )
    z_absolute_penalty = exponential_penalty_function(
        parameter_dict["z_absolute_action_penalty_magnitude"],
        parameter_dict["z_absolute_action_penalty_exponent"],
        action[:, 2],
    )
    yawrate_absolute_penalty = exponential_penalty_function(
        parameter_dict["yawrate_absolute_action_penalty_magnitude"],
        parameter_dict["yawrate_absolute_action_penalty_exponent"],
        action[:, 3],
    )
    absolute_action_penalty = (
        x_absolute_penalty + y_absolute_penalty + z_absolute_penalty + yawrate_absolute_penalty
    )
    return action_diff_penalty + absolute_action_penalty


@torch.jit.script
def _compute_gate_approach_reward(
    robot_position: torch.Tensor,
    gate_position: torch.Tensor,
    parameter_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    # type: (Tensor, Tensor, Dict[str, Tensor]) -> Tensor
    gate_distance = torch.norm(robot_position - gate_position, dim=1)
    return exponential_reward_function(
        parameter_dict["gate_approach_reward_magnitude"],
        0.5,
        gate_distance,
    )


@torch.jit.script
def _compute_camera_facing_reward(
    robot_position: torch.Tensor,
    robot_vehicle_orientation: torch.Tensor,
    gate_position: torch.Tensor,
    parameter_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]
    drone_to_gate = gate_position - robot_position
    drone_to_gate_normalized = drone_to_gate / (
        torch.norm(drone_to_gate, dim=1, keepdim=True) + 1e-8
    )

    qw = robot_vehicle_orientation[:, 3]
    qx = robot_vehicle_orientation[:, 0]
    qy = robot_vehicle_orientation[:, 1]
    qz = robot_vehicle_orientation[:, 2]

    forward_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    forward_y = 2.0 * (qx * qy + qw * qz)
    forward_z = 2.0 * (qx * qz - qw * qy)
    drone_forward = torch.stack([forward_x, forward_y, forward_z], dim=1)
    drone_forward_normalized = drone_forward / (
        torch.norm(drone_forward, dim=1, keepdim=True) + 1e-8
    )

    alignment = torch.sum(drone_forward_normalized * drone_to_gate_normalized, dim=1)
    alignment = torch.clamp(alignment, -1.0, 1.0)

    mag = parameter_dict["camera_facing_reward_magnitude"]
    reward = torch.zeros_like(alignment)

    perfect_mask = alignment > 0.966
    reward[perfect_mask] = mag

    excellent_mask = (alignment > 0.866) & (alignment <= 0.966)
    reward[excellent_mask] = 0.9 * mag * alignment[excellent_mask]

    good_mask = (alignment > 0.5) & (alignment <= 0.866)
    reward[good_mask] = 0.8 * mag * alignment[good_mask]

    moderate_mask = (alignment > 0.0) & (alignment <= 0.5)
    reward[moderate_mask] = 0.4 * mag * alignment[moderate_mask]

    poor_mask = (alignment > -0.707) & (alignment <= 0.0)
    reward[poor_mask] = 0.2 * mag * alignment[poor_mask]

    severe_mask = alignment <= -0.707
    reward[severe_mask] = 2.0 * mag * alignment[severe_mask]

    return reward, alignment


@torch.jit.script
def _compute_gate_alignment_reward(
    robot_position: torch.Tensor,
    gate_position: torch.Tensor,
    gate_width: torch.Tensor,
    gate_height: torch.Tensor,
    gate_center_height: torch.Tensor,
    parameter_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    """Returns (gate_alignment_reward, gate_center_bonus, x_distance_from_center, z_distance_from_center) packed as three tensors."""
    gate_distance = torch.norm(robot_position - gate_position, dim=1)

    gate_width_tolerance = gate_width * 0.6
    aligned_mask = torch.abs(robot_position[:, 0] - gate_position[:, 0]) < gate_width_tolerance
    gate_alignment_reward = torch.zeros_like(gate_distance)
    gate_alignment_reward[aligned_mask] = parameter_dict["gate_alignment_reward_magnitude"]

    x_distance_from_center = torch.abs(robot_position[:, 0] - gate_position[:, 0])
    z_distance_from_center = torch.abs(
        robot_position[:, 2] - (gate_position[:, 2] + gate_center_height)
    )

    x_threshold = gate_width * 0.2
    z_threshold = gate_height * 0.125
    center_aligned_mask = (x_distance_from_center < x_threshold) & (
        z_distance_from_center < z_threshold
    )
    gate_center_bonus = torch.zeros_like(gate_distance)
    gate_center_bonus[center_aligned_mask] = parameter_dict["gate_center_bonus_magnitude"]

    # Pack x/z distances into a 2-column tensor for downstream use
    xz_distances = torch.stack([x_distance_from_center, z_distance_from_center], dim=1)

    return gate_alignment_reward, gate_center_bonus, xz_distances


@torch.jit.script
def _compute_center_passage_bonus(
    just_passed_gate: torch.Tensor,
    x_distance_from_center: torch.Tensor,
    z_distance_from_center: torch.Tensor,
    gate_width: torch.Tensor,
    gate_height: torch.Tensor,
    parameter_dict: dict[str, torch.Tensor],
    gate_distance: torch.Tensor,
) -> torch.Tensor:
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor) -> Tensor
    bonus_mag = parameter_dict["gate_center_passage_bonus_magnitude"]

    x_tol_01 = gate_width * 0.01
    x_tol_02 = gate_width * 0.02
    x_tol_03 = gate_width * 0.03
    x_tol_05 = gate_width * 0.05
    x_tol_07 = gate_width * 0.07
    x_tol_10 = gate_width * 0.10
    x_tol_12 = gate_width * 0.12
    x_tol_15 = gate_width * 0.15
    x_tol_20 = gate_width * 0.20
    z_tol_01 = gate_height * 0.01
    z_tol_02 = gate_height * 0.02
    z_tol_03 = gate_height * 0.03
    z_tol_05 = gate_height * 0.05
    z_tol_07 = gate_height * 0.07
    z_tol_10 = gate_height * 0.10
    z_tol_125 = gate_height * 0.125

    t1 = (
        just_passed_gate & (x_distance_from_center < x_tol_01) & (z_distance_from_center < z_tol_01)
    )
    t2 = (
        just_passed_gate
        & (~t1)
        & (x_distance_from_center < x_tol_02)
        & (z_distance_from_center < z_tol_02)
    )
    t3 = (
        just_passed_gate
        & (~(t1 | t2))
        & (x_distance_from_center < x_tol_03)
        & (z_distance_from_center < z_tol_03)
    )
    t4 = (
        just_passed_gate
        & (~(t1 | t2 | t3))
        & (x_distance_from_center < x_tol_05)
        & (z_distance_from_center < z_tol_05)
    )
    t5 = (
        just_passed_gate
        & (~(t1 | t2 | t3 | t4))
        & (x_distance_from_center < x_tol_07)
        & (z_distance_from_center < z_tol_07)
    )
    t6 = (
        just_passed_gate
        & (~(t1 | t2 | t3 | t4 | t5))
        & (x_distance_from_center < x_tol_10)
        & (z_distance_from_center < z_tol_10)
    )
    t7 = (
        just_passed_gate
        & (~(t1 | t2 | t3 | t4 | t5 | t6))
        & (x_distance_from_center < x_tol_12)
        & (z_distance_from_center < z_tol_10)
    )
    t8 = (
        just_passed_gate
        & (~(t1 | t2 | t3 | t4 | t5 | t6 | t7))
        & (x_distance_from_center < x_tol_15)
        & (z_distance_from_center < z_tol_125)
    )
    t9 = (
        just_passed_gate
        & (~(t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8))
        & (x_distance_from_center < x_tol_20)
        & (z_distance_from_center < z_tol_125)
    )

    bonus = torch.zeros_like(gate_distance)
    bonus[t1] = 1.00 * bonus_mag
    bonus[t2] = 0.92 * bonus_mag
    bonus[t3] = 0.85 * bonus_mag
    bonus[t4] = 0.77 * bonus_mag
    bonus[t5] = 0.65 * bonus_mag
    bonus[t6] = 0.55 * bonus_mag
    bonus[t7] = 0.45 * bonus_mag
    bonus[t8] = 0.35 * bonus_mag
    bonus[t9] = 0.25 * bonus_mag

    return bonus


@torch.jit.script
def _compute_gate_passage_reward(
    robot_position: torch.Tensor,
    gate_position: torch.Tensor,
    gate_passed: torch.Tensor,
    gate_width: torch.Tensor,
    gate_height: torch.Tensor,
    x_distance_from_center: torch.Tensor,
    z_distance_from_center: torch.Tensor,
    parameter_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    """Returns (gate_passage_reward + center_passage_bonus, updated gate_passed, just_passed_gate)."""
    gate_passage_width_tolerance = gate_width * 0.5
    gate_min_height = gate_position[:, 2] + gate_height * 0.0
    gate_max_height = gate_position[:, 2] + gate_height * 1.0

    just_passed_gate = (
        (robot_position[:, 1] > gate_position[:, 1])
        & (torch.abs(robot_position[:, 0] - gate_position[:, 0]) < gate_passage_width_tolerance)
        & (robot_position[:, 2] > gate_min_height)
        & (robot_position[:, 2] < gate_max_height)
        & (~gate_passed)
    )

    gate_distance = torch.norm(robot_position - gate_position, dim=1)
    gate_passage_reward = torch.zeros_like(gate_distance)
    gate_passage_reward[just_passed_gate] = parameter_dict["gate_passage_reward_magnitude"]

    gate_center_passage_bonus = _compute_center_passage_bonus(
        just_passed_gate,
        x_distance_from_center,
        z_distance_from_center,
        gate_width,
        gate_height,
        parameter_dict,
        gate_distance,
    )

    updated_gate_passed = gate_passed | just_passed_gate
    return gate_passage_reward + gate_center_passage_bonus, updated_gate_passed, just_passed_gate


@torch.jit.script
def _compute_altitude_maintenance_reward(
    robot_position: torch.Tensor,
    parameter_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    # type: (Tensor, Dict[str, Tensor]) -> Tensor
    optimal_altitude_min = 1.4
    optimal_altitude_max = 1.6
    current_altitude = robot_position[:, 2]

    altitude_error = torch.zeros_like(current_altitude)
    below_range_mask = current_altitude < optimal_altitude_min
    altitude_error[below_range_mask] = optimal_altitude_min - current_altitude[below_range_mask]
    above_range_mask = current_altitude > optimal_altitude_max
    altitude_error[above_range_mask] = current_altitude[above_range_mask] - optimal_altitude_max

    return exponential_reward_function(
        parameter_dict["altitude_maintenance_reward_magnitude"],
        parameter_dict["altitude_maintenance_reward_exponent"],
        altitude_error,
    )


@torch.jit.script
def _compute_boundary_violation_penalty(
    boundary_violation_one_shot_mask: torch.Tensor,
    parameter_dict: dict[str, torch.Tensor],
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    # type: (Tensor, Dict[str, Tensor], int, Device) -> Tensor
    penalty = torch.zeros(num_envs, device=device)
    penalty[boundary_violation_one_shot_mask] = -parameter_dict[
        "boundary_violation_penalty_magnitude"
    ]
    return penalty


@torch.jit.script
def _compute_collision_penalty(
    reward: torch.Tensor,
    crashes: torch.Tensor,
    parameter_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    # type: (Tensor, Tensor, Dict[str, Tensor]) -> Tensor
    return torch.where(
        crashes > 0,
        parameter_dict["collision_penalty"] * torch.ones_like(reward),
        reward,
    )


@torch.jit.script
def compute_gate_reward(
    pos_error,
    prev_pos_error,
    crashes,
    action,
    prev_action,
    robot_position,
    robot_vehicle_orientation,
    gate_position,
    gate_passed,
    curriculum_progress_fraction,
    parameter_dict,
    gate_width,
    gate_height,
    gate_center_height,
    boundary_violation_one_shot_mask,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]

    MULTIPLICATION_FACTOR_REWARD = 1.0 + (0.5) * curriculum_progress_fraction
    dist = torch.norm(pos_error, dim=1)
    prev_dist_to_goal = torch.norm(prev_pos_error, dim=1)

    pos_reward = _compute_position_reward(dist, parameter_dict)
    very_close_to_goal_reward = _compute_very_close_reward(dist, parameter_dict)
    getting_closer_reward = _compute_getting_closer_reward(dist, prev_dist_to_goal, parameter_dict)
    distance_from_goal_reward = torch.zeros_like(dist)
    total_action_penalty = _compute_action_penalty(action, prev_action, parameter_dict)
    gate_approach_reward = _compute_gate_approach_reward(
        robot_position, gate_position, parameter_dict
    )
    camera_facing_reward, camera_gate_alignment = _compute_camera_facing_reward(
        robot_position, robot_vehicle_orientation, gate_position, parameter_dict
    )
    gate_alignment_reward, gate_center_bonus, xz_distances = _compute_gate_alignment_reward(
        robot_position,
        gate_position,
        gate_width,
        gate_height,
        gate_center_height,
        parameter_dict,
    )
    x_distance_from_center = xz_distances[:, 0]
    z_distance_from_center = xz_distances[:, 1]

    gate_passage_total, gate_passed, just_passed_gate = _compute_gate_passage_reward(
        robot_position,
        gate_position,
        gate_passed,
        gate_width,
        gate_height,
        x_distance_from_center,
        z_distance_from_center,
        parameter_dict,
    )
    altitude_maintenance_reward = _compute_altitude_maintenance_reward(
        robot_position, parameter_dict
    )

    # Gate camera-facing reward: only before first crossing and while approaching gate
    pre_cross_approach = (robot_position[:, 1] < gate_position[:, 1] - 0.1) & (~gate_passed)
    gated_camera_facing = camera_facing_reward * pre_cross_approach.float()

    boundary_violation_penalty = _compute_boundary_violation_penalty(
        boundary_violation_one_shot_mask,
        parameter_dict,
        robot_position.shape[0],
        robot_position.device,
    )

    reward = (
        MULTIPLICATION_FACTOR_REWARD * pos_reward
        + MULTIPLICATION_FACTOR_REWARD * very_close_to_goal_reward
        + MULTIPLICATION_FACTOR_REWARD * getting_closer_reward
        + MULTIPLICATION_FACTOR_REWARD * distance_from_goal_reward
        + MULTIPLICATION_FACTOR_REWARD * gate_approach_reward
        + MULTIPLICATION_FACTOR_REWARD * gate_alignment_reward
        + MULTIPLICATION_FACTOR_REWARD * gate_passage_total
        + MULTIPLICATION_FACTOR_REWARD * gate_center_bonus
        + MULTIPLICATION_FACTOR_REWARD * gated_camera_facing
        + MULTIPLICATION_FACTOR_REWARD * altitude_maintenance_reward
        + boundary_violation_penalty
        + total_action_penalty
    )

    reward = _compute_collision_penalty(reward, crashes, parameter_dict)

    return reward, crashes, camera_gate_alignment
