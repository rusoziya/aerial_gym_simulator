from __future__ import annotations

import torch

from aerial_gym.utils.math import exponential_reward_function, exponential_penalty_function


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
    
    # Base reward computation - REDUCED multiplication factor to prevent over-rewarding
    MULTIPLICATION_FACTOR_REWARD = 1.0 + (0.5) * curriculum_progress_fraction
    dist = torch.norm(pos_error, dim=1)
    prev_dist_to_goal = torch.norm(prev_pos_error, dim=1)
    
    pos_reward = exponential_reward_function(
        parameter_dict["pos_reward_magnitude"],
        parameter_dict["pos_reward_exponent"],
        dist,
    )
    very_close_to_goal_reward = exponential_reward_function(
        parameter_dict["very_close_to_goal_reward_magnitude"],
        parameter_dict["very_close_to_goal_reward_exponent"],
        dist,
    )

    getting_closer = prev_dist_to_goal - dist
    getting_closer_reward = torch.where(
        getting_closer > 0,
        parameter_dict["getting_closer_reward_multiplier"] * getting_closer,
        2.0 * parameter_dict["getting_closer_reward_multiplier"] * getting_closer,
    )

    distance_from_goal_reward = torch.zeros_like(dist)
    
    # Action penalties - FIXED: Added missing Y-action penalties for 4D action space
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
    
    # Absolute action penalties - FIXED: Removed curriculum scaling and added Y-axis penalty
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
    
    absolute_action_penalty = x_absolute_penalty + y_absolute_penalty + z_absolute_penalty + yawrate_absolute_penalty
    total_action_penalty = action_diff_penalty + absolute_action_penalty

    # Gate-specific rewards
    gate_distance = torch.norm(robot_position - gate_position, dim=1)
    
    # Reward for approaching gate
    gate_approach_reward = exponential_reward_function(
        parameter_dict["gate_approach_reward_magnitude"],
        0.5,
        gate_distance,
    )
    
    # Enhanced Camera Facing Reward System - Proportional to alignment angle
    # Calculate vector from drone to gate
    drone_to_gate = gate_position - robot_position
    drone_to_gate_normalized = drone_to_gate / (torch.norm(drone_to_gate, dim=1, keepdim=True) + 1e-8)
    
    # Get drone's forward direction (where camera points)
    # Camera faces forward in drone's body frame (+X direction after orientation)
    # Convert quaternion to rotation matrix and extract forward direction
    qw, qx, qy, qz = robot_vehicle_orientation[:, 3], robot_vehicle_orientation[:, 0], robot_vehicle_orientation[:, 1], robot_vehicle_orientation[:, 2]
    
    # Forward direction in world frame (drone's +X axis)
    forward_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    forward_y = 2.0 * (qx * qy + qw * qz)
    forward_z = 2.0 * (qx * qz - qw * qy)
    drone_forward = torch.stack([forward_x, forward_y, forward_z], dim=1)
    drone_forward_normalized = drone_forward / (torch.norm(drone_forward, dim=1, keepdim=True) + 1e-8)
    
    # Calculate alignment between camera direction and gate direction
    camera_gate_alignment = torch.sum(drone_forward_normalized * drone_to_gate_normalized, dim=1)
    camera_gate_alignment = torch.clamp(camera_gate_alignment, -1.0, 1.0)  # Clamp to [-1, 1]
    
    camera_facing_reward = torch.zeros_like(camera_gate_alignment)
    
    # PERFECT ALIGNMENT: 0-15° (alignment > 0.966) - Maximum reward
    perfect_mask = camera_gate_alignment > 0.966  # cos(15°) ≈ 0.966
    camera_facing_reward[perfect_mask] = parameter_dict["camera_facing_reward_magnitude"]  # Full 5.0 reward
    
    # EXCELLENT ALIGNMENT: 15-30° (0.866 < alignment ≤ 0.966) - High reward
    excellent_mask = (camera_gate_alignment > 0.866) & (camera_gate_alignment <= 0.966)  # cos(30°) = 0.866
    camera_facing_reward[excellent_mask] = 0.9 * parameter_dict["camera_facing_reward_magnitude"] * camera_gate_alignment[excellent_mask]
    
    # GOOD ALIGNMENT: 30-60° (0.5 < alignment ≤ 0.866) - High reward  
    good_mask = (camera_gate_alignment > 0.5) & (camera_gate_alignment <= 0.866)  # cos(60°) = 0.5
    camera_facing_reward[good_mask] = 0.8 * parameter_dict["camera_facing_reward_magnitude"] * camera_gate_alignment[good_mask]
    
    # MODERATE ALIGNMENT: 60-90° (0 < alignment ≤ 0.5) - Moderate reward
    moderate_mask = (camera_gate_alignment > 0.0) & (camera_gate_alignment <= 0.5)
    camera_facing_reward[moderate_mask] = 0.4 * parameter_dict["camera_facing_reward_magnitude"] * camera_gate_alignment[moderate_mask]
    
    # POOR ALIGNMENT: 90-135° (-0.707 < alignment ≤ 0) - Small penalty
    poor_mask = (camera_gate_alignment > -0.707) & (camera_gate_alignment <= 0.0)  # cos(135°) ≈ -0.707
    camera_facing_reward[poor_mask] = 0.2 * parameter_dict["camera_facing_reward_magnitude"] * camera_gate_alignment[poor_mask]  # Small penalty
    
    # SEVERELY MISALIGNED: 135-180° (alignment ≤ -0.707) - Strong penalty
    severe_mask = camera_gate_alignment <= -0.707
    camera_facing_reward[severe_mask] = 2.0 * parameter_dict["camera_facing_reward_magnitude"] * camera_gate_alignment[severe_mask]  # Strong penalty
    
    # Reward for gate alignment (being in front of gate opening)
    gate_alignment_reward = torch.zeros_like(gate_distance)
    # Check if robot is roughly aligned with gate opening (Y direction) - ADAPTIVE to gate width
    gate_width_tolerance = gate_width * 0.6  # 60% of gate width for alignment tolerance
    aligned_mask = torch.abs(robot_position[:, 0] - gate_position[:, 0]) < gate_width_tolerance
    gate_alignment_reward[aligned_mask] = parameter_dict["gate_alignment_reward_magnitude"]
    
    # Enhanced center alignment rewards for precise gate navigation - ADAPTIVE to gate size
    gate_center_bonus = torch.zeros_like(gate_distance)
    # Distance from gate center in X direction (horizontal alignment)
    x_distance_from_center = torch.abs(robot_position[:, 0] - gate_position[:, 0])
    # Distance from gate center in Z direction (vertical alignment) - ADAPTIVE to gate center height
    z_distance_from_center = torch.abs(robot_position[:, 2] - (gate_position[:, 2] + gate_center_height))
    
    # Check if robot is very close to gate center - ADAPTIVE thresholds
    x_threshold = gate_width * 0.2  # 20% of gate width for precise X alignment
    z_threshold = gate_height * 0.125  # 12.5% of gate height for precise Z alignment
    center_aligned_mask = (x_distance_from_center < x_threshold) & (z_distance_from_center < z_threshold)
    gate_center_bonus[center_aligned_mask] = parameter_dict["gate_center_bonus_magnitude"]
    
    # Check for gate passage (crossing Y = 0 plane with proper alignment) - ADAPTIVE to gate dimensions
    # Passage window covers the entire gate opening
    gate_passage_width_tolerance = gate_width * 0.5  # half-width
    gate_min_height = gate_position[:, 2] + gate_height * 0.0
    gate_max_height = gate_position[:, 2] + gate_height * 1.0
    
    just_passed_gate = (
        (robot_position[:, 1] > gate_position[:, 1]) &  # In front of gate
        (torch.abs(robot_position[:, 0] - gate_position[:, 0]) < gate_passage_width_tolerance) &  # Within gate width
        (robot_position[:, 2] > gate_min_height) & (robot_position[:, 2] < gate_max_height) &  # Within gate height
        (~gate_passed)  # Haven't passed before
    )
    
    # Center passage bonus: piecewise tiers by proximity to gate center (ADAPTIVE)
    gate_passage_reward = torch.zeros_like(gate_distance)
    gate_passage_reward[just_passed_gate] = parameter_dict["gate_passage_reward_magnitude"]
    
    gate_center_passage_bonus = torch.zeros_like(gate_distance)
    bonus_mag = parameter_dict["gate_center_passage_bonus_magnitude"]

    # Tolerances (as fractions of gate size) — denser piecewise
    x_tol_01  = gate_width  * 0.01
    x_tol_02  = gate_width  * 0.02
    x_tol_03  = gate_width  * 0.03
    x_tol_05  = gate_width  * 0.05
    x_tol_07  = gate_width  * 0.07
    x_tol_10  = gate_width  * 0.10
    x_tol_12  = gate_width  * 0.12
    x_tol_15  = gate_width  * 0.15
    x_tol_20  = gate_width  * 0.20
    z_tol_01  = gate_height * 0.01
    z_tol_02  = gate_height * 0.02
    z_tol_03  = gate_height * 0.03
    z_tol_05  = gate_height * 0.05
    z_tol_07  = gate_height * 0.07
    z_tol_10  = gate_height * 0.10
    z_tol_125 = gate_height * 0.125

    # Define tier masks (mutually exclusive, most strict first)
    t1 =  just_passed_gate & (x_distance_from_center < x_tol_01) & (z_distance_from_center < z_tol_01)
    t2 =  just_passed_gate & (~t1) & (x_distance_from_center < x_tol_02) & (z_distance_from_center < z_tol_02)
    t3 =  just_passed_gate & (~(t1 | t2)) & (x_distance_from_center < x_tol_03) & (z_distance_from_center < z_tol_03)
    t4 =  just_passed_gate & (~(t1 | t2 | t3)) & (x_distance_from_center < x_tol_05) & (z_distance_from_center < z_tol_05)
    t5 =  just_passed_gate & (~(t1 | t2 | t3 | t4)) & (x_distance_from_center < x_tol_07) & (z_distance_from_center < z_tol_07)
    t6 =  just_passed_gate & (~(t1 | t2 | t3 | t4 | t5)) & (x_distance_from_center < x_tol_10) & (z_distance_from_center < z_tol_10)
    t7 =  just_passed_gate & (~(t1 | t2 | t3 | t4 | t5 | t6)) & (x_distance_from_center < x_tol_12) & (z_distance_from_center < z_tol_10)
    t8 =  just_passed_gate & (~(t1 | t2 | t3 | t4 | t5 | t6 | t7)) & (x_distance_from_center < x_tol_15) & (z_distance_from_center < z_tol_125)
    t9 =  just_passed_gate & (~(t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8)) & (x_distance_from_center < x_tol_20) & (z_distance_from_center < z_tol_125)

    # Assign piecewise bonuses (fractions of bonus_mag)
    gate_center_passage_bonus[t1] = 1.00 * bonus_mag
    gate_center_passage_bonus[t2] = 0.92 * bonus_mag
    gate_center_passage_bonus[t3] = 0.85 * bonus_mag
    gate_center_passage_bonus[t4] = 0.77 * bonus_mag
    gate_center_passage_bonus[t5] = 0.65 * bonus_mag
    gate_center_passage_bonus[t6] = 0.55 * bonus_mag
    gate_center_passage_bonus[t7] = 0.45 * bonus_mag
    gate_center_passage_bonus[t8] = 0.35 * bonus_mag
    gate_center_passage_bonus[t9] = 0.25 * bonus_mag
    
    # Update gate passed status
    gate_passed = gate_passed | just_passed_gate

    optimal_altitude_min = 1.4  # meters
    optimal_altitude_max = 1.6  # meters  
    current_altitude = robot_position[:, 2]
    
    # Calculate distance from optimal altitude range
    altitude_error = torch.zeros_like(current_altitude)
    # Below optimal range
    below_range_mask = current_altitude < optimal_altitude_min
    altitude_error[below_range_mask] = optimal_altitude_min - current_altitude[below_range_mask]
    # Above optimal range  
    above_range_mask = current_altitude > optimal_altitude_max
    altitude_error[above_range_mask] = current_altitude[above_range_mask] - optimal_altitude_max
    # Within optimal range - no error
    
    # Exponential reward for being at optimal altitude
    altitude_maintenance_reward = exponential_reward_function(
        parameter_dict["altitude_maintenance_reward_magnitude"],
        parameter_dict["altitude_maintenance_reward_exponent"],
        altitude_error,
    )

    # Calculate individual component contributions (for debugging)
    multiplied_pos_reward = MULTIPLICATION_FACTOR_REWARD * pos_reward
    multiplied_very_close_reward = MULTIPLICATION_FACTOR_REWARD * very_close_to_goal_reward  
    multiplied_getting_closer = MULTIPLICATION_FACTOR_REWARD * getting_closer_reward
    multiplied_distance_reward = MULTIPLICATION_FACTOR_REWARD * distance_from_goal_reward
    multiplied_gate_approach = MULTIPLICATION_FACTOR_REWARD * gate_approach_reward
    multiplied_gate_alignment = MULTIPLICATION_FACTOR_REWARD * gate_alignment_reward
    multiplied_gate_passage = MULTIPLICATION_FACTOR_REWARD * gate_passage_reward
    multiplied_gate_center_bonus = MULTIPLICATION_FACTOR_REWARD * gate_center_bonus
    multiplied_gate_center_passage = MULTIPLICATION_FACTOR_REWARD * gate_center_passage_bonus
    # Gate the camera-facing reward in TorchScript path as well: only before first crossing and while approaching gate
    pre_cross_approach = (robot_position[:, 1] < gate_position[:, 1] - 0.1) & (~gate_passed)
    gated_camera_facing = camera_facing_reward * pre_cross_approach.float()
    multiplied_camera_facing = MULTIPLICATION_FACTOR_REWARD * gated_camera_facing
    multiplied_altitude_maintenance = MULTIPLICATION_FACTOR_REWARD * altitude_maintenance_reward

    # Boundary violation penalty: one-shot mask computed in Python to avoid repeated penalties
    boundary_violation_penalty = torch.zeros_like(gate_distance)
    boundary_violation_penalty[boundary_violation_one_shot_mask] = -parameter_dict["boundary_violation_penalty_magnitude"]

    # Combined reward - NOW INCLUDING CAMERA FACING REWARD AND ALTITUDE MAINTENANCE
    reward = (
        multiplied_pos_reward
        + multiplied_very_close_reward
        + multiplied_getting_closer
        + multiplied_distance_reward
        + multiplied_gate_approach
        + multiplied_gate_alignment
        + multiplied_gate_passage
        + multiplied_gate_center_bonus
        + multiplied_gate_center_passage
        + multiplied_camera_facing  # Camera facing reward (gated pre-crossing)
        + multiplied_altitude_maintenance  # NEW: Altitude maintenance reward
        + boundary_violation_penalty  # Penalty for crossing behind gate plane without a valid passage
        + total_action_penalty
    )

    # Apply collision penalties
    reward[:] = torch.where(
        crashes > 0,
        parameter_dict["collision_penalty"] * torch.ones_like(reward),
        reward,
    )
    
    return reward, crashes, camera_gate_alignment