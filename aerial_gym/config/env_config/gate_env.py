from aerial_gym.config.asset_config.gate_asset_config import gate_asset_params
from aerial_gym.config.asset_config.env_asset_config import (
    tree_asset_params, 
    TREE_SEMANTIC_ID, 
    WALL_SEMANTIC_ID,
    left_wall,
    right_wall,
    front_wall,
    back_wall,
    bottom_wall
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
        
        # Simulation timing
        num_physics_steps_per_env_step_mean = 1  # Steps between renders
        num_physics_steps_per_env_step_std = 0   # No variation in timing
        render_viewer_every_n_steps = 1  # Render every step for smooth visualization
        
        # Collision and reset behavior
        collision_force_threshold = 0.02  # Lower threshold for more sensitive collision detection [N]
        reset_on_collision = True  # Reset environment when drone hits gate
        
        # Ground and basic setup
        create_ground_plane = True  # Create ground for visual reference
        
        # Observation and latency simulation
        sample_timestep_for_latency = False  # No latency simulation for clean testing
        perturb_observations = False  # No observation noise for clean testing
        keep_same_env_for_num_episodes = 1  # Reset environments each episode
        write_to_sim_at_every_timestep = False
        
        # Rendering backend
        use_warp = False  # Use Isaac Gym rendering for simplicity
        
        # Environment bounds - match the working example exactly
        # Gate will be at center, with plenty of room to approach and exit
        lower_bound_min = [-8.0, -8.0, 0.0]  # Ground level starts at Z=0 (like working example)
        lower_bound_max = [-8.0, -8.0, 0.0]  # Fixed bounds (no randomization)
        upper_bound_min = [8.0, 8.0, 8.0]    # Match working example bounds exactly
        upper_bound_max = [8.0, 8.0, 8.0]    # Fixed bounds (no randomization)
    
    class env_config:
        """
        Configuration specifying which assets to include in the environment.
        
        This environment includes:
        - gate: The main gate structure for flying through
        - trees: Background trees for visual richness and depth perception
        """
        
        # Asset inclusion configuration
        include_asset_type = {
            "gate": True,  # Include the gate asset
            "trees": True,  # Include trees for background
            "left_wall": True,  # Left boundary wall
            "right_wall": True,  # Right boundary wall
            "front_wall": True,  # Front boundary wall
            "back_wall": True,  # Back boundary wall
            "bottom_wall": True,  # Ground plane
            # "top_wall": False,  # Keep top open for flying
        }
        
        # Custom tree configuration for gate environment
        class gate_tree_params(tree_asset_params):
            num_assets = 5  # Add 5 trees for good background coverage
            
            # ENABLE COLLISION for trees so they appear in segmentation!
            collision_mask = 0  # 0 = enable collision, 1 = disable collision
            
            # Position trees behind the gate using specific coordinates (gate is rotated 90°)
            # Trees should be at positive Y coordinates (behind gate from camera perspective)
            # Environment bounds are typically [-8, 8] so we convert absolute coords to ratios
            # Tree positions from working example: (-2.0, 3.0, 0.0), (0.0, 4.0, 0.0), (2.0, 3.0, 0.0), (-1.0, 5.0, 0.0), (1.0, 5.0, 0.0)
            # Convert to ratios: position_ratio = (position + 8) / 16 for range [-8, 8]
            
            # For simplicity, position trees in a line behind the gate
            min_state_ratio = [
                0.3, 0.7, 0.0,  # X: left side, Y: behind gate (positive Y), Z: on ground
                0, -np.pi/6.0, -np.pi,  # Rotation
                1.0,  # Scale
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Velocities
            ]
            max_state_ratio = [
                0.7, 0.8, 0.0,  # X: right side, Y: further behind gate, Z: on ground
                0, np.pi/6.0, np.pi,  # Rotation
                1.0,  # Scale
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Velocities
            ]
            
            # Trees should stay in environment and be fixed
            keep_in_env = True
            fix_base_link = True
            collapse_fixed_joints = True
            color = [70, 200, 100]  # Green trees
            
            # CRITICAL: Add the same segmentation properties as the gate!
            semantic_id = TREE_SEMANTIC_ID  # Use the defined tree semantic ID
            body_semantic_label = TREE_SEMANTIC_ID  # Body-level semantic labeling
            link_semantic_label = TREE_SEMANTIC_ID  # Link-level semantic labeling
            per_link_semantic = True  # Enable per-link semantic labeling (like gate)
            semantic_masked_links = {}  # No masked links, all branches should be segmented
            
            # Physical properties for proper collision detection (like gate)
            disable_gravity = True  # Trees are fixed in place
            density = 1000.0  # Realistic tree density for collision
            
            # Add force sensor for collision detection (like gate)
            place_force_sensor = True  # Enable force sensor to detect collisions
            force_sensor_parent_link = "branch_0"  # Attach to main trunk (first branch)
        
        # Custom wall configurations with collision and segmentation enabled
        class boundary_left_wall(left_wall):
            collision_mask = 0  # Enable collision detection
            semantic_id = WALL_SEMANTIC_ID
            body_semantic_label = WALL_SEMANTIC_ID
            link_semantic_label = WALL_SEMANTIC_ID
            per_link_semantic = True
            place_force_sensor = True
            color = [150, 150, 150]  # Gray walls
            
        class boundary_right_wall(right_wall):
            collision_mask = 0  # Enable collision detection
            semantic_id = WALL_SEMANTIC_ID
            body_semantic_label = WALL_SEMANTIC_ID
            link_semantic_label = WALL_SEMANTIC_ID
            per_link_semantic = True
            place_force_sensor = True
            color = [150, 150, 150]  # Gray walls
            
        class boundary_front_wall(front_wall):
            collision_mask = 0  # Enable collision detection
            semantic_id = WALL_SEMANTIC_ID
            body_semantic_label = WALL_SEMANTIC_ID
            link_semantic_label = WALL_SEMANTIC_ID
            per_link_semantic = True
            place_force_sensor = True
            color = [150, 150, 150]  # Gray walls
            
        class boundary_back_wall(back_wall):
            collision_mask = 0  # Enable collision detection
            semantic_id = WALL_SEMANTIC_ID
            body_semantic_label = WALL_SEMANTIC_ID
            link_semantic_label = WALL_SEMANTIC_ID
            per_link_semantic = True
            place_force_sensor = True
            color = [150, 150, 150]  # Gray walls
            
        class boundary_bottom_wall(bottom_wall):
            collision_mask = 0  # Enable collision detection (ground plane)
            semantic_id = WALL_SEMANTIC_ID
            body_semantic_label = WALL_SEMANTIC_ID
            link_semantic_label = WALL_SEMANTIC_ID
            per_link_semantic = True
            place_force_sensor = True
            color = [100, 80, 60]  # Brown ground
        
        # Mapping of asset type names to their configuration classes
        asset_type_to_dict_map = {
            "gate": gate_asset_params,
            "trees": gate_tree_params,
            "left_wall": boundary_left_wall,
            "right_wall": boundary_right_wall,
            "front_wall": boundary_front_wall,
            "back_wall": boundary_back_wall,
            "bottom_wall": boundary_bottom_wall,
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
            tree_asset_params,
            object_asset_params
        )
        
        # Include gate plus some obstacles
        include_asset_type = {
            "gate": True,
            "trees": True,    # Add some trees around the gate
            "objects": False, # Could enable for even more challenge
        }
        
        # Configure obstacles to be sparse and not block the gate
        class sparse_tree_params(tree_asset_params):
            num_assets = 2  # Just a couple of trees
            
            # ENABLE COLLISION for trees so they appear in segmentation!
            collision_mask = 0  # 0 = enable collision, 1 = disable collision
            
            # Position trees away from center where gate is located
            min_state_ratio = [0.1, 0.1, 0.0, 0, -np.pi/6.0, -np.pi, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            max_state_ratio = [0.9, 0.9, 0.0, 0, np.pi/6.0, np.pi, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            # CRITICAL: Add the same segmentation properties as the gate!
            semantic_id = TREE_SEMANTIC_ID  # Use the defined tree semantic ID
            body_semantic_label = TREE_SEMANTIC_ID  # Body-level semantic labeling
            link_semantic_label = TREE_SEMANTIC_ID  # Link-level semantic labeling
            per_link_semantic = True  # Enable per-link semantic labeling (like gate)
            semantic_masked_links = {}  # No masked links, all branches should be segmented
            
            # Physical properties for proper collision detection (like gate)
            disable_gravity = True  # Trees are fixed in place
            density = 1000.0  # Realistic tree density for collision
            
            # Add force sensor for collision detection (like gate)
            place_force_sensor = True  # Enable force sensor to detect collisions
            force_sensor_parent_link = "branch_0"  # Attach to main trunk (first branch)
        
        # Asset mapping
        asset_type_to_dict_map = {
            "gate": gate_asset_params,
            "trees": sparse_tree_params,
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