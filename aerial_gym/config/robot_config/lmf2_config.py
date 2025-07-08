import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY

from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)
from aerial_gym.config.sensor_config.lidar_config.osdome_64_config import OSDome_64_Config
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig


class LMF2Cfg:

    class init_config:
        # SPAWN NEAR STATIC CAMERA FACING GATE DIRECTLY
        # Static camera at (0, -3.0, 1.0) behind gate, facing towards gate (yaw = 0)
        # Gate is rotated 90° around Z-axis, so opening faces +X direction
        # Environment bounds: [-4, 4] x [-4, 4] x [0, 4]
        # Position conversion: ratio = (position + 4) / 8 for X,Y; ratio = position / 4 for Z
        # 
        # Static camera position (0, -3.0, 1.0) = ratios (0.5, 0.125, 0.25)
        # Small variations: ±0.3m in X, ±0.2m in Y, ±0.2m in Z
        # FACE GATE OPENING DIRECTLY: yaw = π/2 (facing +X towards gate opening) with NO variation
        
        min_init_state = [
            0.4625, # ratio_x: X = -0.3 → ratio = ((-0.3) + 4) / 8 = 0.4625 (slight left of camera)
            0.1,    # ratio_y: Y = -3.2 → ratio = ((-3.2) + 4) / 8 = 0.1 (slightly behind camera)
            0.35,   # ratio_z: Z = 1.4 → ratio = 1.4 / 4 = 0.35 (AT GATE LEVEL - INCREASED from 0.2)
            0,      # no roll
            0,      # no pitch  
            np.pi/2, # yaw: face directly towards gate opening (+X direction, 90°)
            1.0,
            -0.1,   # very small initial velocity variation
            -0.1,
            0.1,    # POSITIVE initial Z-velocity to help gain altitude (CHANGED from -0.1)
            -0.1,
            -0.1,
            -0.1,
        ]
        max_init_state = [
            0.5375, # ratio_x: X = +0.3 → ratio = ((+0.3) + 4) / 8 = 0.5375 (slight right of camera)
            0.15,   # ratio_y: Y = -2.8 → ratio = ((-2.8) + 4) / 8 = 0.15 (slightly in front of camera)
            0.4,    # ratio_z: Z = 1.6 → ratio = 1.6 / 4 = 0.4 (OPTIMAL GATE HEIGHT - INCREASED from 0.3)
            0,      # no roll
            0,      # no pitch
            np.pi/2, # yaw: face directly towards gate opening (+X direction, 90°)
            1.0,
            0.1,    # very small initial velocity variation
            0.1,
            0.2,    # POSITIVE initial Z-velocity to help gain altitude (CHANGED from 0.1)
            0.1,
            0.1,
            0.1,
        ]

    class sensor_config:
        enable_camera = True
        camera_config = BaseDepthCameraConfig

        enable_lidar = False
        lidar_config = BaseLidarConfig  # OSDome_64_Config

        enable_imu = False
        imu_config = BaseImuConfig

    class disturbance:
        enable_disturbance = True
        prob_apply_disturbance = 0.05
        max_force_and_torque_disturbance = [4.75, 4.75, 4.75, 0.03, 0.03, 0.03]

    class damping:
        linvel_linear_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        angular_linear_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes

    class robot_asset:
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/lmf2"
        file = "model.urdf"
        name = "base_quadrotor"  # actor name
        base_link_name = "base_link"
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints.
        fix_base_link = False  # fix the base of the robot
        collision_mask = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up
        density = 0.000001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.001

        semantic_id = 0
        per_link_semantic = False

        min_state_ratio = [
            0.4625, # ratio_x: X = -0.3 → ratio = ((-0.3) + 4) / 8 = 0.4625 (slight left of camera)
            0.1,    # ratio_y: Y = -3.2 → ratio = ((-3.2) + 4) / 8 = 0.1 (slightly behind camera)
            0.35,   # ratio_z: Z = 1.4 → ratio = 1.4 / 4 = 0.35 (AT GATE LEVEL - INCREASED from 0.2)
            0,      # no roll
            0,      # no pitch
            np.pi/2, # yaw: face directly towards gate opening (+X direction, 90°)
            1.0,
            0,      # no initial velocity
            0,
            0,      # no initial velocity
            0,
            0,
            0,
        ]  # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
        max_state_ratio = [
            0.5375, # ratio_x: X = +0.3 → ratio = ((+0.3) + 4) / 8 = 0.5375 (slight right of camera)
            0.15,   # ratio_y: Y = -2.8 → ratio = ((-2.8) + 4) / 8 = 0.15 (slightly in front of camera)
            0.4,    # ratio_z: Z = 1.6 → ratio = 1.6 / 4 = 0.4 (OPTIMAL GATE HEIGHT - INCREASED from 0.3)
            0,      # no roll
            0,      # no pitch
            np.pi/2, # yaw: face directly towards gate opening (+X direction, 90°)
            1.0,
            0,      # no initial velocity
            0,
            0,      # no initial velocity
            0,
            0,
            0,
        ]

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
        force_application_level = "base_link"  # "motor_link" or "root_link" decides to apply combined forces acting on the robot at the root link or at the individual motor links

        application_mask = [1 + 4 + i for i in range(0, 4)]
        motor_directions = [1, -1, 1, -1]

        allocation_matrix = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [-0.13, -0.13, 0.13, 0.13],
            [-0.13, 0.13, 0.13, -0.13],
            [-0.07, 0.07, -0.07, 0.07],
        ]

        class motor_model_config:
            use_rps = True
            motor_thrust_constant_min = 0.00000926312
            motor_thrust_constant_max = 0.00001826312
            motor_time_constant_increasing_min = 0.05
            motor_time_constant_increasing_max = 0.08
            motor_time_constant_decreasing_min = 0.005
            motor_time_constant_decreasing_max = 0.005
            max_thrust = 15.0
            min_thrust = 0.1
            max_thrust_rate = 100000.0
            thrust_to_torque_ratio = 0.07
            use_discrete_approximation = True  # use discrete approximation for motor dynamics


# Configuration for Drone 1 (positioned at -2, 0, 1.5)
class LMF2Drone1Cfg(LMF2Cfg):
    class init_config(LMF2Cfg.init_config):
        # Position drone 1 at (-2, 0, 1.5) - left side of center
        min_init_state = [
            0.3,  # ratio_x: -2.0 in [-5, 5] range = ((-2 + 5) / 10) = 0.3
            0.5,  # ratio_y: 0.0 in [-5, 5] range = ((0 + 5) / 10) = 0.5 
            0.625,  # ratio_z: 1.5 in [-1, 3] range = ((1.5 + 1) / 4) = 0.625
            0,  # no rotation
            0,  # no rotation 
            0,  # no rotation
            1.0,
            0,  # no velocity
            0,  # no velocity
            0,  # no velocity
            0,  # no angular velocity
            0,  # no angular velocity
            0,  # no angular velocity
        ]
        max_init_state = [
            0.3,  # same as min for fixed position
            0.5,  # same as min for fixed position
            0.625,  # same as min for fixed position
            0,  # no rotation
            0,  # no rotation
            0,  # no rotation
            1.0,
            0,  # no velocity
            0,  # no velocity
            0,  # no velocity
            0,  # no angular velocity
            0,  # no angular velocity
            0,  # no angular velocity
        ]
    
    class robot_asset(LMF2Cfg.robot_asset):
        name = "drone_1"  # unique name for first drone
        semantic_id = 1  # different semantic ID for identification


# Configuration for Drone 2 (positioned at 2, 0, 1.5)
class LMF2Drone2Cfg(LMF2Cfg):
    class init_config(LMF2Cfg.init_config):
        # Position drone 2 at (2, 0, 1.5) - right side of center
        min_init_state = [
            0.7,  # ratio_x: 2.0 in [-5, 5] range = ((2 + 5) / 10) = 0.7
            0.5,  # ratio_y: 0.0 in [-5, 5] range = ((0 + 5) / 10) = 0.5
            0.625,  # ratio_z: 1.5 in [-1, 3] range = ((1.5 + 1) / 4) = 0.625
            0,  # no rotation
            0,  # no rotation
            0,  # no rotation
            1.0,
            0,  # no velocity
            0,  # no velocity
            0,  # no velocity
            0,  # no angular velocity
            0,  # no angular velocity
            0,  # no angular velocity
        ]
        max_init_state = [
            0.7,  # same as min for fixed position
            0.5,  # same as min for fixed position
            0.625,  # same as min for fixed position
            0,  # no rotation
            0,  # no rotation
            0,  # no rotation
            1.0,
            0,  # no velocity
            0,  # no velocity
            0,  # no velocity
            0,  # no angular velocity
            0,  # no angular velocity
            0,  # no angular velocity
        ]
    
    class robot_asset(LMF2Cfg.robot_asset):
        name = "drone_2"  # unique name for second drone
        semantic_id = 2  # different semantic ID for identification
