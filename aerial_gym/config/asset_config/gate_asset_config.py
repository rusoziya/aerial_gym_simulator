from aerial_gym.config.asset_config.base_asset import BaseAssetParams
from aerial_gym import AERIAL_GYM_DIRECTORY
import numpy as np

# Semantic ID for gate objects
GATE_SEMANTIC_ID = 10


class GateAssetConfig:
    class gate_asset_params(BaseAssetParams):
        """
        Configuration for gate assets that drones can fly through.
        Creates a rectangular gate structure with two vertical posts and a horizontal beam.
        """
        num_assets = 1  # Number of gates to spawn in the environment
        
        # Asset file location
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"
        file = "gate.urdf"  # Specific URDF file to use
        
        # Position ratios within environment bounds
        # Center the gate in the environment
        # Environment Z bounds are now [0.0, 8.0], so for Z=0 (ground level): ratio = 0.0
        min_position_ratio = [0.5, 0.5, 0.0]  # Center XY, exactly at ground level Z=0
        max_position_ratio = [0.5, 0.5, 0.0]  # Fixed position
        
        # Alternatively, can randomize gate position (uncomment to use):
        # min_position_ratio = [0.3, 0.3, 0.0]  # Allow some randomization
        # max_position_ratio = [0.7, 0.7, 0.0]  # Within central area
        
        # Orientation - rotate gate 90 degrees around Z-axis so trees are behind it
        min_euler_angles = [0.0, 0.0, np.pi/2]  # 90° rotation around Z-axis
        max_euler_angles = [0.0, 0.0, np.pi/2]  # Fixed orientation
        
        # For randomized orientation, use:
        # min_euler_angles = [0.0, 0.0, -np.pi/4]  # Rotate up to ±45 degrees
        # max_euler_angles = [0.0, 0.0, np.pi/4]
        
        # Full state specification (pos_x, pos_y, pos_z, roll, pitch, yaw, scale, vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z)
        min_state_ratio = [
            0.5, 0.5, 0.0,  # Position (center of environment, exactly at ground level Z=0)
            0.0, 0.0, np.pi/2,  # Orientation (90° rotation around Z-axis)
            1.0,            # Scale (no scaling)
            0.0, 0.0, 0.0,  # Linear velocity (stationary)
            0.0, 0.0, 0.0   # Angular velocity (stationary)
        ]
        max_state_ratio = [
            0.5, 0.5, 0.0,  # Same as min for fixed position
            0.0, 0.0, np.pi/2,  # Same as min for fixed orientation
            1.0,            # Same scale
            0.0, 0.0, 0.0,  # Stationary
            0.0, 0.0, 0.0   # Stationary
        ]
        
        # Physics properties for collision detection
        collision_mask = 0  # Enable collision detection (0 = enable, 1 = disable)
        disable_gravity = True  # Disable gravity since gate is fixed in place
        fix_base_link = True  # Gate is stationary/fixed in place
        
        # Simulation optimization
        replace_cylinder_with_capsule = True  # Faster collision detection with capsules
        collapse_fixed_joints = True  # Optimize rigid body structure
        
        # Physical properties for realistic collision response
        density = 2000.0  # Heavy material (concrete/steel) for realistic collisions
        angular_damping = 1.0  # High damping to prevent unwanted motion
        linear_damping = 1.0   # High damping to keep gate stationary
        max_angular_velocity = 0.0  # No rotation allowed (fixed structure)
        max_linear_velocity = 0.0   # No movement allowed (fixed structure)
        
        # Visual properties
        color = [150, 150, 150]  # Gray color for the gate
        
        # Semantic labeling
        body_semantic_label = GATE_SEMANTIC_ID
        link_semantic_label = GATE_SEMANTIC_ID
        per_link_semantic = True  # Each part of gate has semantic label
        semantic_id = GATE_SEMANTIC_ID
        
        # Environment interaction
        keep_in_env = True  # Gate stays in environment (reverted from False)
        
        # Sensor configuration for collision detection
        place_force_sensor = True  # Enable force sensor to detect collisions with drone
        force_sensor_parent_link = "base_link"  # Attach sensor to gate base
        
        # No collision mesh override needed
        use_collision_mesh_instead_of_visual = False


# Create instances that can be imported
gate_asset_params = GateAssetConfig.gate_asset_params 