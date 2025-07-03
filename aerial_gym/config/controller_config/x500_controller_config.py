import numpy as np


class control:
    """
    Control parameters optimized for X500 quadrotor
    Based on X500 specifications:
    - Motor thrust constant: 8.54858e-6
    - Max thrust per motor: 20.0N (total 80N)
    - Standard X-configuration quadrotor
    - Larger frame than LMF2, requires different gains
    
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
    max_inclination_angle_rad = np.pi / 3.0  # Same as other drones
    max_yaw_rate = np.pi / 3.0  # Same as other drones

    # X500-specific position gains (lower than LMF2 due to heavier mass)
    K_pos_tensor_max = [1.8, 1.8, 1.2]  # Reduced from LMF2's [2.0, 2.0, 1.0]
    K_pos_tensor_min = [1.5, 1.5, 0.8]  # Reduced from LMF2's [2.0, 2.0, 1.0]

    # X500-specific velocity gains (higher than LMF2 due to more mass/inertia)
    K_vel_tensor_max = [
        4.0,  # Increased from LMF2's 3.3 (more damping needed)
        4.0,  # Increased from LMF2's 3.3
        2.0,  # Increased from LMF2's 1.3 (vertical needs more damping)
    ]
    K_vel_tensor_min = [3.2, 3.2, 1.6]  # Increased from LMF2's [2.7, 2.7, 1.7]

    # X500-specific rotation gains (adjusted for different inertia characteristics)
    K_rot_tensor_max = [
        2.2,   # Increased from LMF2's 1.85 (larger frame, more inertia)
        2.2,   # Increased from LMF2's 1.85
        0.6,   # Increased from LMF2's 0.4 (yaw inertia)
    ]
    K_rot_tensor_min = [1.8, 1.8, 0.4]  # Increased from LMF2's [1.6, 1.6, 0.25]

    # X500-specific angular velocity gains (higher due to larger inertia)
    K_angvel_tensor_max = [
        0.8,   # Increased from LMF2's 0.5 (more angular damping needed)
        0.8,   # Increased from LMF2's 0.5
        0.15,  # Increased from LMF2's 0.09 (yaw damping)
    ]
    K_angvel_tensor_min = [0.6, 0.6, 0.12]  # Increased from LMF2's [0.4, 0.4, 0.075]

    randomize_params = True  # Enable parameter randomization for robust training 