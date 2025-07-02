import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY

from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.config.sensor_config.camera_config.d455_depth_config import (
    RsD455Config,
)
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)
from aerial_gym.config.sensor_config.lidar_config.osdome_64_config import OSDome_64_Config
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig


class X500Cfg:

    class init_config:
        # init_state tensor is of the format [ratio_x, ratio_y, ratio_z, roll_radians, pitch_radians, yaw_radians, 1.0 (for maintaining shape), vx, vy, vz, wx, wy, wz]
        # Static camera position: (0.0, -3.0, 1.5) in world coordinates
        # Converting to ratios for gate environment bounds: x∈[-8,8], y∈[-8,8], z∈[0,8]
        # ratio_x = (0 - (-8)) / (8 - (-8)) = 8/16 = 0.5
        # ratio_y = (-3 - (-8)) / (8 - (-8)) = 5/16 = 0.3125
        # ratio_z = (1.5 - 0) / (8 - 0) = 1.5/8 = 0.1875
        min_init_state = [
            0.5,      # ratio_x: spawn at x=0 (static camera X position)
            0.3125,   # ratio_y: spawn at y=-3 (static camera Y position)  
            0.1875,   # ratio_z: spawn at z=1.5 (static camera Z position)
            0,        # roll: no rotation
            0,        # pitch: no rotation 
            0,        # yaw: facing forward (same as static camera direction)
            1.0,
            0,        # vx: no initial velocity
            0,        # vy: no initial velocity
            0,        # vz: no initial velocity
            0,        # wx: no angular velocity
            0,        # wy: no angular velocity
            0,        # wz: no angular velocity
        ]
        max_init_state = [
            0.5,      # same as min for fixed position
            0.3125,   # same as min for fixed position
            0.1875,   # same as min for fixed position
            0,        # no rotation
            0,        # no rotation
            0,        # no rotation 
            1.0,
            0,        # no initial velocity
            0,        # no initial velocity
            0,        # no initial velocity
            0,        # no angular velocity
            0,        # no angular velocity
            0,        # no angular velocity
        ]

    class sensor_config:
        enable_camera = True  # Enable camera for D455 depth camera
        camera_config = RsD455Config  # Use Intel RealSense D455 configuration

        enable_lidar = False
        lidar_config = BaseLidarConfig  # OSDome_64_Config

        enable_imu = False
        imu_config = BaseImuConfig

    class disturbance:
        enable_disturbance = False
        prob_apply_disturbance = 0.00
        max_force_and_torque_disturbance = [0, 0, 0, 0, 0, 0]  # [fx, fy, fz, tx, ty, tz]

    class damping:
        linvel_linear_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        angular_linear_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes

    class robot_asset:
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/x500"
        file = "model.urdf"
        name = "base_quadrotor"  # actor name
        base_link_name = "base_link"
        disable_gravity = False
        collapse_fixed_joints = False  # merge bodies connected by fixed joints.
        fix_base_link = False  # fix the base of the robot
        collision_mask = 0  # 0 to enable collision detection, 1 to disable
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up
        density = 0.000001
        angular_damping = 0.02
        linear_damping = 0.02
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.00001

        semantic_id = 0
        per_link_semantic = False

        min_state_ratio = [
            0.1,
            0.1,
            0.1,
            0,
            0,
            -np.pi,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
        max_state_ratio = [
            0.9,
            0.9,
            0.9,
            0,
            0,
            np.pi,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]

        max_force_and_torque_disturbance = [
            0.1,
            0.1,
            0.1,
            0.05,
            0.05,
            0.05,
        ]  # [fx, fy, fz, tx, ty, tz]

        color = None
        semantic_masked_links = {}
        keep_in_env = True  # this does nothing for the robot

        min_position_ratio = None
        max_position_ratio = None

        min_euler_angles = [-np.pi, -np.pi, -np.pi]
        max_euler_angles = [np.pi, np.pi, np.pi]

        place_force_sensor = True  # set this to True if IMU is desired
        force_sensor_parent_link = "base_link"
        force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # [x, y, z, qx, qy, qz, qw]

        use_collision_mesh_instead_of_visual = False  # does nothing for the robot

    class control_allocator_config:
        num_motors = 4
        force_application_level = "motor_link"  # "motor_link" or "root_link" decides to apply combined forces acting on the robot at the root link or at the individual motor links

        application_mask = [4, 1, 3, 2] # front right, back_left, front_left, back_right
        motor_directions = [1, 1, -1, -1]

        allocation_matrix = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [-0.13, 0.13, 0.13, -0.13],
            [-0.13, 0.13, -0.13, 0.13],
            [-0.025, 0.025, -0.025, 0.025],
        ]

        class motor_model_config:
            use_rps = True
            motor_thrust_constant_min = 8.54858e-6 #0.00000926312
            motor_thrust_constant_max = 8.54858e-6 #0.00001826312
            motor_time_constant_increasing_min = 0.0125
            motor_time_constant_increasing_max = 0.0125
            motor_time_constant_decreasing_min = 0.025
            motor_time_constant_decreasing_max = 0.025
            max_thrust = 20.0
            min_thrust = 0.0
            max_thrust_rate = 100000.0
            thrust_to_torque_ratio = 0.025
            use_discrete_approximation = False  # use discrete approximation for motor dynamics
            max_thrust = 20.0
            min_thrust = 0.0
            max_thrust_rate = 100000.0
            thrust_to_torque_ratio = 0.025
            use_discrete_approximation = False  # use discrete approximation for motor dynamics