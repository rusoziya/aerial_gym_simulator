from aerial_gym.config.asset_config.gate_asset_config import gate_asset_params
from aerial_gym.config.asset_config.env_asset_config import (
    object_asset_params, 
    OBJECT_SEMANTIC_ID, 
    WALL_SEMANTIC_ID,
    left_wall,
    right_wall,
    front_wall,
    back_wall,
    bottom_wall,
    top_wall
)
import numpy as np


class GateEnvCfg:
    """
    Environment configuration for a gate flying scenario.
    
    This environment contains:
    - A single gate structure that drones can fly through
    - Open space around the gate for maneuvering
    - Minimal obstacles to focus on gate navigation
    """
    
    class env:
        # Basic environment parameters
        num_envs = 1  # Number of parallel environments (default to 1 for simplicity)
        num_env_actions = 0  # No environment-controlled actions (gate is static)
        env_spacing = 10.0  # Space between parallel environments
        
        # MATCH ORIGINAL: Simulation timing and physics steps
        num_physics_steps_per_env_step_mean = 10  # MATCH ORIGINAL: 10 steps between renders (was 1)
        num_physics_steps_per_env_step_std = 0   # MATCH ORIGINAL: No variation in timing
        render_viewer_every_n_steps = 1  # Render every step for smooth visualization
        
        # MATCH ORIGINAL: Collision and reset behavior
        collision_force_threshold = 0.05  # MATCH ORIGINAL: 50mN threshold (was 0.02 = 20mN)
        reset_on_collision = True  # Reset environment when drone hits gate
        
        # MATCH ORIGINAL: Ground and basic setup
        create_ground_plane = False  # MATCH ORIGINAL: No ground plane (was True)
        
        # MATCH ORIGINAL: Observation and latency simulation
        sample_timestep_for_latency = True  # MATCH ORIGINAL: Enable latency simulation (was False)
        perturb_observations = True  # MATCH ORIGINAL: Enable observation noise (was False)
        keep_same_env_for_num_episodes = 1  # Reset environments each episode
        write_to_sim_at_every_timestep = False  # MATCH ORIGINAL: Don't write every timestep
        
        # Rendering backend
        use_warp = True  # MATCH ORIGINAL: Use Warp for consistent rendering and lighting
        
        # Environment bounds - 8x8x4 environment for focused navigation (4 unit radius)
        # Gate will be at center, with adequate room to approach and exit
        lower_bound_min = [-4.0, -4.0, 0.0]  # Ground level starts at Z=0
        lower_bound_max = [-4.0, -4.0, 0.0]  # Fixed bounds (no randomization)
        upper_bound_min = [4.0, 4.0, 4.0]    # 8x8x4 environment bounds  
        upper_bound_max = [4.0, 4.0, 4.0]    # Fixed bounds (no randomization)
    
    class env_config:
        """
        Configuration specifying which assets to include in the environment.
        
        This environment includes:
        - gate: The main gate structure for flying through
        - objects: Random obstacles for visual richness and navigation challenge
        """
        
        # Asset inclusion configuration
        include_asset_type = {
            "gate": True,  # Include the gate asset
            "objects": True,  # Include random objects for obstacles
            "left_wall": True,  # Left boundary wall
            "right_wall": True,  # Right boundary wall
            "front_wall": True,  # Front boundary wall
            "back_wall": True,  # Back boundary wall
            "bottom_wall": True,  # Ground plane
            "top_wall": True,  # Top wall to close the environment
        }
        
        # Custom random object configuration for gate environment
        class gate_object_params(object_asset_params):
            num_assets = 3  # Balanced obstacle count for gate navigation (matches curriculum range)
            
            # ENABLE COLLISION for objects so they appear in segmentation!
            collision_mask = 0  # 0 = enable collision, 1 = disable collision
            
            # Position objects behind the gate using specific coordinates (gate is rotated 90°)
            # Objects should be at positive Y coordinates (behind gate from camera perspective)
            # Environment bounds are now [-4, 4] so we convert absolute coords to ratios
            # Convert to ratios: position_ratio = (position + 4) / 8 for range [-4, 4]
            # Objects positioned on obstacle side (positive Y) where drone shouldn't spawn
            
            # For 8x8 environment, position objects in a line behind the gate
            min_state_ratio = [
                0.1, 0.65, 0.0,  # X: wider spread from left, Y: behind gate (positive Y), Z: on ground
                0, -np.pi/3.0, -np.pi,  # More rotation variation
                0.6,  # Smaller minimum scale 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Velocities
            ]
            max_state_ratio = [
                0.9, 0.95, 0.3,  # X: wider spread to right, Y: well behind gate, Z: higher elevation
                0, np.pi/3.0, np.pi,  # More rotation variation
                1.5,  # Larger maximum scale for more visual diversity
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Velocities
            ]
            
            # Objects should stay in environment and be fixed
            keep_in_env = True
            fix_base_link = True
            collapse_fixed_joints = True
            color = [100, 150, 200]  # Blue-gray objects
            
            # CRITICAL: Add the same segmentation properties as the gate!
            semantic_id = OBJECT_SEMANTIC_ID
            body_semantic_label = OBJECT_SEMANTIC_ID  # Body-level semantic labeling
            link_semantic_label = OBJECT_SEMANTIC_ID  # Link-level semantic labeling
            per_link_semantic = True  # Enable per-link semantic labeling (like gate)
            semantic_masked_links = {}  # No masked links, all parts should be segmented
            
            # Physical properties for proper collision detection (like gate)
            disable_gravity = True  # Objects are fixed in place
            density = 1000.0  # Realistic object density for collision
            
            # Add force sensor for collision detection (like gate)
            place_force_sensor = True  # Enable force sensor to detect collisions
            force_sensor_parent_link = "base_link"  # Attach to main object
        
        # MATCH ORIGINAL: Use same wall colors, heights, and positions as original environment
        class boundary_left_wall(left_wall):
            collision_mask = 0  # Enable collision detection
            semantic_id = WALL_SEMANTIC_ID
            body_semantic_label = WALL_SEMANTIC_ID
            link_semantic_label = WALL_SEMANTIC_ID
            per_link_semantic = True
            place_force_sensor = True
            color = [100, 200, 210]  # MATCH ORIGINAL: Teal walls (was [150, 150, 150])
            keep_in_env = True  # Keep in environment (reverted from False)
            
            # MATCH ORIGINAL: Full height walls like original environment
            min_state_ratio = [
                0.5, 1.0, 0.5,  # MATCH ORIGINAL: Position at Z=0.5 (was 0.4), full height
                0.0, 0.0, 0.0,  # Orientation
                1.0,            # MATCH ORIGINAL: Full scale (was 0.6)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Velocities
            ]
            max_state_ratio = [
                0.5, 1.0, 0.5,  # Same position
                0.0, 0.0, 0.0,  # Same orientation
                1.0,            # Same full scale
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Velocities
            ]
            
        class boundary_right_wall(right_wall):
            collision_mask = 0  # Enable collision detection
            semantic_id = WALL_SEMANTIC_ID
            body_semantic_label = WALL_SEMANTIC_ID
            link_semantic_label = WALL_SEMANTIC_ID
            per_link_semantic = True
            place_force_sensor = True
            color = [100, 200, 210]  # MATCH ORIGINAL: Teal walls (was [150, 150, 150])
            keep_in_env = True  # Keep in environment (reverted from False)
            
            # MATCH ORIGINAL: Full height walls like original environment
            min_state_ratio = [
                0.5, 0.0, 0.5,  # MATCH ORIGINAL: Position at Z=0.5 (was 0.4), full height
                0.0, 0.0, 0.0,  # Orientation
                1.0,            # MATCH ORIGINAL: Full scale (was 0.6)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Velocities
            ]
            max_state_ratio = [
                0.5, 0.0, 0.5,  # Same position
                0.0, 0.0, 0.0,  # Same orientation
                1.0,            # Same full scale
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Velocities
            ]
            
        class boundary_front_wall(front_wall):
            collision_mask = 0  # Enable collision detection
            semantic_id = WALL_SEMANTIC_ID
            body_semantic_label = WALL_SEMANTIC_ID
            link_semantic_label = WALL_SEMANTIC_ID
            per_link_semantic = True
            place_force_sensor = True
            color = [100, 200, 210]  # MATCH ORIGINAL: Teal walls (was [150, 150, 150])
            keep_in_env = True  # Keep in environment (reverted from False)
            
            # MATCH ORIGINAL: Full height walls like original environment
            min_state_ratio = [
                1.0, 0.5, 0.5,  # MATCH ORIGINAL: Position at Z=0.5 (was 0.4), full height
                0.0, 0.0, 0.0,  # Orientation
                1.0,            # MATCH ORIGINAL: Full scale (was 0.6)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Velocities
            ]
            max_state_ratio = [
                1.0, 0.5, 0.5,  # Same position
                0.0, 0.0, 0.0,  # Same orientation
                1.0,            # Same full scale
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Velocities
            ]
            
        class boundary_back_wall(back_wall):
            collision_mask = 0  # Enable collision detection
            semantic_id = WALL_SEMANTIC_ID
            body_semantic_label = WALL_SEMANTIC_ID
            link_semantic_label = WALL_SEMANTIC_ID
            per_link_semantic = True
            place_force_sensor = True
            color = [100, 200, 210]  # MATCH ORIGINAL: Teal walls (was [150, 150, 150])
            keep_in_env = True  # Keep in environment (reverted from False)
            
            # MATCH ORIGINAL: Full height walls like original environment
            min_state_ratio = [
                0.0, 0.5, 0.5,  # MATCH ORIGINAL: Position at Z=0.5 (was 0.4), full height
                0.0, 0.0, 0.0,  # Orientation
                1.0,            # MATCH ORIGINAL: Full scale (was 0.6)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Velocities
            ]
            max_state_ratio = [
                0.0, 0.5, 0.5,  # Same position
                0.0, 0.0, 0.0,  # Same orientation
                1.0,            # Same full scale
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Velocities
            ]
            
        class boundary_bottom_wall(bottom_wall):
            collision_mask = 0  # Enable collision detection (ground plane)
            semantic_id = WALL_SEMANTIC_ID
            body_semantic_label = WALL_SEMANTIC_ID
            link_semantic_label = WALL_SEMANTIC_ID
            per_link_semantic = True
            place_force_sensor = True
            color = [100, 150, 150]  # MATCH ORIGINAL: Darker teal ground (was [100, 80, 60])
            keep_in_env = True  # Keep in environment (reverted from False)
        
        class boundary_top_wall(top_wall):
            collision_mask = 0  # Enable collision detection
            semantic_id = WALL_SEMANTIC_ID
            body_semantic_label = WALL_SEMANTIC_ID
            link_semantic_label = WALL_SEMANTIC_ID
            per_link_semantic = True
            place_force_sensor = True
            color = [100, 200, 210]  # MATCH ORIGINAL: Teal ceiling (was [120, 120, 120])
            keep_in_env = True  # Keep in environment (reverted from False)
            
            # MATCH ORIGINAL: Keep top wall at full height like original environment
            min_state_ratio = [
                0.5, 0.5, 1.0,  # MATCH ORIGINAL: Position at Z=1.0 (full height)
                0.0, 0.0, 0.0,  # Orientation
                1.0,            # MATCH ORIGINAL: Full scale
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Velocities
            ]
            max_state_ratio = [
                0.5, 0.5, 1.0,  # Same position
                0.0, 0.0, 0.0,  # Same orientation
                1.0,            # Same full scale
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Velocities
            ]
        
        # Mapping of asset type names to their configuration classes
        asset_type_to_dict_map = {
            "gate": gate_asset_params,
            "objects": gate_object_params,
            "left_wall": boundary_left_wall,
            "right_wall": boundary_right_wall,
            "front_wall": boundary_front_wall,
            "back_wall": boundary_back_wall,
            "bottom_wall": boundary_bottom_wall,
            "top_wall": boundary_top_wall,
        }


class GateEnvWithObstaclesCfg(GateEnvCfg):
    """
    Extended gate environment with additional obstacles for more challenging navigation.
    
    This variant adds some obstacles around the gate to make the flight more challenging
    while still focusing on the gate traversal task.
    """
    
    class env(GateEnvCfg.env):
        # Make environment slightly more challenging
        num_envs = 8  # Fewer parallel environments due to added complexity
        collision_force_threshold = 0.03  # More sensitive collision detection
    
    class env_config:
        # Import additional obstacle configurations if needed
        from aerial_gym.config.asset_config.env_asset_config import (
            object_asset_params
        )
        
        # Include gate plus some obstacles
        include_asset_type = {
            "gate": True,
            "objects": False, # Could enable for even more challenge
        }
        
        # Configure obstacles to be sparse and not block the gate
        class sparse_object_params(object_asset_params):
            num_assets = 2  # Just a couple of objects
            
            # ENABLE COLLISION for objects so they appear in segmentation!
            collision_mask = 0  # 0 = enable collision, 1 = disable collision
            
            # Position objects away from center where gate is located
            min_state_ratio = [0.1, 0.1, 0.0, 0, -np.pi/6.0, -np.pi, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            max_state_ratio = [0.9, 0.9, 0.0, 0, np.pi/6.0, np.pi, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            # CRITICAL: Add the same segmentation properties as the gate!
            semantic_id = OBJECT_SEMANTIC_ID  # Use the defined object semantic ID
            body_semantic_label = OBJECT_SEMANTIC_ID  # Body-level semantic labeling
            link_semantic_label = OBJECT_SEMANTIC_ID  # Link-level semantic labeling
            per_link_semantic = True  # Enable per-link semantic labeling (like gate)
            semantic_masked_links = {}  # No masked links, all parts should be segmented
            
            # Physical properties for proper collision detection (like gate)
            disable_gravity = True  # Objects are fixed in place
            density = 1000.0  # Realistic object density for collision
            
            # Add force sensor for collision detection (like gate)
            place_force_sensor = True  # Enable force sensor to detect collisions
            force_sensor_parent_link = "base_link"  # Attach to main object
        
        # Asset mapping
        asset_type_to_dict_map = {
            "gate": gate_asset_params,
            "objects": sparse_object_params,
        }


class GateEnvRandomizedCfg(GateEnvCfg):
    """
    Gate environment with randomized gate position and orientation.
    
    This variant randomizes the gate placement to train more robust navigation policies.
    """
    
    class env_config:
        # Create a randomized gate configuration
        class randomized_gate_params(gate_asset_params):
            # Randomize gate position within central area
            min_position_ratio = [0.3, 0.3, 0.0]
            max_position_ratio = [0.7, 0.7, 0.0]
            
            # Randomize gate orientation
            min_euler_angles = [0.0, 0.0, -np.pi/3]  # Up to ±60 degrees rotation
            max_euler_angles = [0.0, 0.0, np.pi/3]
            
            # Update state ratios to match position/orientation randomization
            min_state_ratio = [
                0.3, 0.3, 0.0,           # Position
                0.0, 0.0, -np.pi/3,      # Orientation
                1.0,                     # Scale
                0.0, 0.0, 0.0,          # Linear velocity
                0.0, 0.0, 0.0           # Angular velocity
            ]
            max_state_ratio = [
                0.7, 0.7, 0.0,           # Position
                0.0, 0.0, np.pi/3,       # Orientation
                1.0,                     # Scale
                0.0, 0.0, 0.0,          # Linear velocity
                0.0, 0.0, 0.0           # Angular velocity
            ]
        
        # Asset inclusion
        include_asset_type = {
            "gate": True,
        }
        
        # Asset mapping with randomized gate
        asset_type_to_dict_map = {
            "gate": randomized_gate_params,
        } 