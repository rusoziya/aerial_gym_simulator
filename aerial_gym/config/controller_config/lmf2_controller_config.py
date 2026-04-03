from __future__ import annotations

import numpy as np


class control:
    """
    Control parameters
    controller:
        lee_position_control: command_actions = [x, y, z, yaw] in environment frame scaled between -1 and 1
        lee_velocity_control: command_actions = [vx, vy, vz, yaw_rate] in vehicle frame scaled between -1 and 1
        lee_attitude_control: command_actions = [thrust, roll, pitch, yaw_rate] in vehicle frame scaled between -1 and 1
    kP: gains for position
    kV: gains for velocity
    kR: gains for attitude
    kOmega: gains for angular velocity
    """

    num_actions = 4
    max_inclination_angle_rad = np.pi / 3.0
    # max_yaw_rate = 1.5
    max_yaw_rate = np.pi / 3.0
    K_pos_tensor_max = [
        2.2,
        2.2,
        2.8,
    ]  # INCREASED Z from 1.0 to 2.5, added 20% variation for domain randomization
    K_pos_tensor_min = [1.8, 1.8, 2.2]  # Added 20% variation range for position gain randomization

    # ENHANCED VELOCITY DAMPING for improved velocity controller stability
    K_vel_tensor_max = [
        4.5,  # INCREASED from 3.3 for stronger X-Y damping (velocity controller stability)
        4.5,  # INCREASED from 3.3 for stronger X-Y damping (velocity controller stability)
        5.0,  # INCREASED from 4.0 for stronger Z damping (altitude stability)
    ]  # used for lee_position_control, lee_velocity_control only

    K_vel_tensor_min = [
        3.8,
        3.8,
        4.2,
    ]  # INCREASED from [2.7, 2.7, 3.5] for consistent stronger damping

    K_rot_tensor_max = [
        1.85,
        1.85,
        0.4,
    ]  # used for lee_position_control, lee_velocity_control and lee_attitude_control
    K_rot_tensor_min = [1.6, 1.6, 0.25]

    K_angvel_tensor_max = [
        0.5,
        0.5,
        0.09,
    ]  # used for lee_position_control, lee_velocity_control and lee_attitude_control
    K_angvel_tensor_min = [0.4, 0.4, 0.075]

    randomize_params = True
