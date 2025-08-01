import torch
from aerial_gym import AERIAL_GYM_DIRECTORY


class task_config:
    seed = -1
    sim_name = "base_sim"
    env_name = "gate_env"  # Use gate environment instead of env_with_obstacles
    robot_name = "lmf2"  # Use proven LMF2 robot
    controller_name = "lmf2_velocity_control"  # CHANGED: Switch to velocity controller for more direct control
    args = {}
    num_envs = 16  # Standard configuration for gate navigation training
    use_warp = True
    headless = False  # Enable visualization to view both cameras (can be overridden by Sample Factory)
    device = "cuda:0"
    
    # Enhanced observation space: 3D drone position + 6D static camera pose + 3D full orientation + 10D state + 64D drone VAE + 64D static camera VAE = 150D
    # MODIFIED: Added drone absolute position (3D) and full yaw sensing (3D orientation instead of 2D)
    # Previous: 6D static camera pose + 2D orientation + 9D state + 128D camera = 147D  
    # Current:  3D drone position + 6D static camera pose + 3D full orientation + 10D state + 128D camera = 150D (position-aware navigation)
    # State breakdown: 3D linear velocity + 3D angular velocity + 4D actions = 10D
    # Both cameras now share the same VAE model to reduce GPU memory usage by ~50%
    observation_space_dim = 150  # Enhanced to include drone position and full orientation = 150D (3+6+3+10+64+64)
    privileged_observation_space_dim = 0
    action_space_dim = 4  # 4D action space [x_vel, y_vel, z_vel, yaw_rate] for velocity control
    episode_len_steps = 100  # REDUCED: Faster episodes for quicker training feedback and evaluation

    return_state_before_reset = (
        False  # False as usually state is returned for next episode after reset
    )
    # user can set the above to true if they so desire

    # Target positions (goals) - keep targets on front side of gate
    # Front side (positive Y) where obstacles are, forcing gate navigation
    # Obstacles at Y = [+2.0, +3.2], so targets should be beyond Y = +3.6
    # FIXED: Target Z-range within gate flyable zone (0.2-2.2m → ratios 0.05-0.55)
    target_min_ratio = [0.2, 0.95, 0.35]  # Y=0.95 -> Y=+3.6 (beyond obstacles), Z=0.35 -> Z=1.4m (gate level)
    target_max_ratio = [0.8, 0.99, 0.65]  # Y=0.99 -> Y=+3.92 (well beyond obstacles), Z=0.45 -> Z=1.8m (within gate)
    
    # GATE DIMENSIONS ANALYSIS (from gate.urdf):
    # - Gate opening: 2.5m wide (Y = ±1.25m) × 2.3m tall (Z = 0.1m to 2.4m)
    # - Usable flight space: Z = 0.2m to 2.2m (safe margins from gate structure)
    # - Environment bounds: [-4,4] x [-4,4] x [0,4] = 8m × 8m × 4m
    # 
    # GATE ORIENTATION (CRITICAL) - UPDATED FOR 8x8x4 ENVIRONMENT:
    # Gate rotated 90° around Z-axis, so opening faces Y-direction
    # Original: posts at ±Y, opening along X → After rotation: posts at ±X, opening along Y
    # 
    # COORDINATE SYSTEM (Environment bounds: [-4,4] x [-4,4] x [0,4]):
    # - Gate at center (0,0,0) with opening facing Y-direction
    # - Objects/obstacles at Y ratios 0.75-0.9 = Y positions +2.0 to +3.2 (FRONT of gate)
    # - Static camera at Y = -3.0 (BEHIND gate)
    # - Drone must approach from BEHIND gate (negative Y) to fly through to FRONT
    #
    # RATIO TO POSITION CONVERSION: position = ratio * 8 - 4 (for X,Y), position = ratio * 4 (for Z)
    # - Y ratio 0.0 → Y = -4.0 (far behind gate)
    # - Y ratio 0.5 → Y = 0.0 (AT gate center)
    # - Y ratio 0.75 → Y = +2.0 (obstacle start)  
    # - Y ratio 1.0 → Y = +4.0 (far in front)
    
    # Enhanced reward parameters for gate navigation with 4D action space
    reward_parameters = {
        # BASE NAVIGATION REWARDS (SCALED DOWN to prevent reward inflation)
        "pos_reward_magnitude": 2.5,  # REDUCED from 5.0 (10x reduction)
        "pos_reward_exponent": 1.0 / 3.5,
        "very_close_to_goal_reward_magnitude": 2.5,  # REDUCED from 5.0 (10x reduction)
        "very_close_to_goal_reward_exponent": 2.0,
        "getting_closer_reward_multiplier": 5.0,  # REDUCED from 10.0 (10x reduction)
        
        # Action smoothness penalties (match base navigation)
        "x_action_diff_penalty_magnitude": 0.8,
        "x_action_diff_penalty_exponent": 3.333,
        "y_action_diff_penalty_magnitude": 0.8,  # Y-axis smoothness penalty  
        "y_action_diff_penalty_exponent": 3.333,
        # "z_action_diff_penalty_magnitude": 0.8,  # REDUCED from 0.8 to allow necessary altitude adjustments
        # "z_action_diff_penalty_exponent": 5.0,  # REDUCED from 3.333 for gentler Z-smoothness
        "z_action_diff_penalty_magnitude": 0.4,  # REDUCED from 0.8 to allow necessary altitude adjustments
        "z_action_diff_penalty_exponent": 2.0,  # REDUCED from 3.333 for gentler Z-smoothness

        # "yawrate_action_diff_penalty_magnitude": 0.8,
        # "yawrate_action_diff_penalty_exponent": 3.33,
        "yawrate_action_diff_penalty_magnitude": 0.5,  # REDUCED from 0.8 for less restrictive yaw control
        "yawrate_action_diff_penalty_exponent": 2.5,  # REDUCED from 3.33 for less restrictive yaw control
        
        # Action magnitude penalties (match base navigation for crash prevention)
        "x_absolute_action_penalty_magnitude": 0.1,
        "x_absolute_action_penalty_exponent": 0.3,
        "y_absolute_action_penalty_magnitude": 0.1,  # Y-axis action penalty
        "y_absolute_action_penalty_exponent": 0.3,
        # "z_absolute_action_penalty_magnitude": 1.5,  # Strong vertical movement penalty
        # "z_absolute_action_penalty_exponent": 1.0,
        # "z_absolute_action_penalty_magnitude": 0.1,  # REDUCED from 0.1 to allow necessary Z-movements
        # "z_absolute_action_penalty_exponent": 0.3,   # REDUCED from 0.3 for even gentler Z-penalty
        "z_absolute_action_penalty_magnitude": 0.05,  # REDUCED from 0.1 to allow necessary Z-movements
        "z_absolute_action_penalty_exponent": 0.2,   # REDUCED from 0.3 for even gentler Z-penalty
        # "yawrate_absolute_action_penalty_magnitude": 1.5,
        # "yawrate_absolute_action_penalty_exponent": 2.0,
        "yawrate_absolute_action_penalty_magnitude": 1.0,  # REDUCED from 1.5 for less restrictive yaw control
        "yawrate_absolute_action_penalty_exponent": 1.5,  # REDUCED from 2.0 for less restrictive yaw control
        
        # CRITICAL: Strong collision penalty to prevent ground crashes
        "collision_penalty": -100.0,  # Match base navigation - STRONG penalty for any collision
        
        # GATE-SPECIFIC REWARDS (SCALED DOWN to prevent reward inflation)
        "gate_approach_reward_magnitude": 1.25,  # REDUCED from 5.0 (10x reduction)
        # "gate_approach_reward_magnitude": 2.0,  # OLD value
        "gate_alignment_reward_magnitude": 0.5,  # REDUCED from 2.0 (10x reduction)
        # "gate_alignment_reward_magnitude": 1.0,  # OLD value
        "gate_passage_reward_magnitude": 50.0,  # INCREASED from 10.0 (one-time bonus should be high)
        "gate_center_bonus_magnitude": 1.25,  # REDUCED from 5.0 (10x reduction)
        # "gate_center_passage_bonus_magnitude": 15.0,  # OLD value
        "gate_center_passage_bonus_magnitude": 100.0,  # INCREASED from 50.0 (one-time center bonus should be high)
        # "camera_facing_reward_magnitude": 5.0,  # Enhanced reward for drone camera facing towards gate (from user's previous request)
        "camera_facing_reward_magnitude": 0.0,  # Enhanced reward for drone camera facing towards gate (TRIPLED from 1.0)

        
        # NEW: Altitude maintenance reward to encourage proper gate-level flying
        "altitude_maintenance_reward_magnitude": 0.0,  # Reward for staying at gate height (1.2-1.8m)
        "altitude_maintenance_reward_exponent": 0.0,   # Exponential reward for being at optimal altitude
        
        # Gate collision penalty (separate from general collision for specificity)
        "gate_collision_penalty": -50.0,  # Additional penalty for hitting gate specifically
    }

    # Shared VAE configuration for both drone and static cameras (Memory-Optimized)
    # Using a single VAE model reduces GPU memory usage by ~50% compared to separate models
    class vae_config:
        use_vae = True
        latent_dims = 64
        model_file = (
            AERIAL_GYM_DIRECTORY
            + "/aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"
        )
        model_folder = AERIAL_GYM_DIRECTORY
        image_res = (270, 480)
        interpolation_mode = "nearest"
        return_sampled_latent = True

    # ===== NEW MULTI-ASPECT CURRICULUM LEARNING SYSTEM =====
    # Comprehensive domain randomization with multiple difficulty aspects
    class curriculum:
        # EXPANDED CURRICULUM RANGE: Level 3-23 (20 levels total)
        min_level = 3   # Start with 3 obstacles behind gate (base level)
        max_level = 23  # End with maximum difficulty (20 levels of progression)
        
        # ===== 5. CAMERA NOISE PROGRESSION (Levels 3-23) - D455 Realistic Noise =====
        # Simulate Intel RealSense D455 camera characteristics for both drone and static cameras
        enable_camera_noise = True                # Enable curriculum-dependent camera noise
        camera_noise_start_level = 3             # Start minimal noise from level 3 
        camera_noise_end_level = 23              # Reach maximum noise at level 23
        max_gaussian_noise_std = 0.0125          # Maximum Gaussian noise: 1.25% of depth range
        max_pixel_dropout_rate = 0.025           # Maximum pixel dropout: 2.5% of pixels
        
        # EVALUATION PARAMETERS
        check_after_log_instances = 128  # INCREASED FREQUENCY: Check curriculum every 128 instances for faster progression
        increase_step = 1  # Increase by 1 level at a time for fine-grained progression
        decrease_step = 0   # NO DECREASE POLICY: Once a level is reached, never go back
        success_rate_for_increase = 0.0   # 1% success rate to progress to next curriculum level (very aggressive progression)
        success_rate_for_decrease = 0.0   # DISABLED: Never decrease difficulty (no-decrease policy)
        
        # MULTI-ASPECT DIFFICULTY PROGRESSION
        # Each curriculum level controls multiple aspects of difficulty:
        
        # 1. OBSTACLE COUNT PROGRESSION (Levels 3-23: Increase obstacles behind gate)
        # Levels 3-23: Direct 1:1 mapping - curriculum level = obstacle count
        # Level 3: 3 obstacles, Level 4: 4 obstacles, ..., Level 23: 23 obstacles
        max_obstacles_behind_gate = 25  # Maximum obstacles behind gate (level 23 + buffer)
        
        # CRITICAL FIX: Ensure asset capacity matches environment configuration  
        # Gate environment now loads 30 objects initially, curriculum can use up to 25
        total_asset_capacity = 30  # Must match gate_object_params.num_assets in gate_env.py
        
        # 2. DRONE SPAWNING: FIXED PARAMETERS (NO CURRICULUM DEPENDENCY)
        # Spawn parameters are now fixed in LMF2 robot config:
        # - Lateral position: ±0.5m variation around gate center (2m behind gate)
        # - Orientation: ±45° randomization (fixed, not curriculum-dependent)
        # - Initial velocity: Minimal random values for additional variation
        
        # 3. STATIC CAMERA DIFFICULTY PROGRESSION (Levels 5-23)
        # Levels 3-4: Camera directly behind gate facing forward (easiest)
        # Levels 5-12: Camera at side angles but still seeing full gate
        # Levels 13-20: Camera at extreme angles with partial gate visibility
        # Levels 21-23: Camera at hardest angles with minimal gate visibility
        camera_orientation_start_level = 5    # Start camera orientation changes at level 5
        max_camera_angle_degrees = 25         # Maximum camera angle from straight-on view (was 30)
        
        # DEBUGGING AND MONITORING
        enable_detailed_logging = True  # Enable comprehensive curriculum debugging
        log_curriculum_changes = True   # Log all curriculum aspect changes
        save_curriculum_metrics = True  # Save curriculum metrics to wandb

        def update_curriculim_level(self, success_rate, current_level):
            """
            ENHANCED CURRICULUM UPDATE WITH NO-DECREASE POLICY
            
            This function implements a no-decrease policy where once a level is reached,
            the difficulty never goes back down. This ensures consistent progression
            and prevents oscillation between difficulty levels.
            """
            # ONLY ALLOW INCREASES (No-decrease policy)
            if success_rate > self.success_rate_for_increase:
                new_level = min(current_level + self.increase_step, self.max_level)
                return new_level
            else:
                # Maintain current level (never decrease)
                return current_level
        
        @staticmethod
        def get_obstacle_count_behind_gate(level):
            """
            Calculate number of obstacles behind gate based on curriculum level.
            
            FIXED: Direct 1:1 mapping between curriculum level and obstacle count
            - Level 3: 3 obstacles
            - Level 4: 4 obstacles
            - ...
            - Level 23: 23 obstacles
            
            Simple progression: obstacle_count = curriculum_level
            """
            max_obstacles_behind_gate = 25  # INCREASED: Support up to 25 obstacles (level 23 + buffer)
            total_asset_capacity = 30  # Must match gate_object_params.num_assets in gate_env.py
            
            # SIMPLE DIRECT MAPPING: curriculum level = obstacle count
            requested_obstacles = level
            
            # CRITICAL VALIDATION: Ensure we never exceed total asset capacity
            if requested_obstacles > total_asset_capacity:
                print(f"WARNING: Curriculum requested {requested_obstacles} obstacles but only {total_asset_capacity} available!")
                requested_obstacles = total_asset_capacity
            
            return requested_obstacles
        
        @staticmethod
        def get_camera_noise(level):
            """
            Calculate camera noise parameters based on curriculum level.
            
            LINEAR PROGRESSION: Level 3 → Level 23
            - Level 3: 0% noise (no noise at start)
            - Level 23: Maximum noise (full D455 simulation)
            - Linear interpolation between levels
            
            D455 Camera Noise Simulation:
            - Gaussian noise: Simulates depth measurement uncertainty  
            - Pixel dropouts: Simulates missing depth readings
            
            Args:
                level: Current curriculum level (3-23)
                
            Returns:
                tuple: (gaussian_std, dropout_rate) - Noise parameters for current level
            """
            # Linear progression constants
            camera_noise_start_level = 3       # Start at level 3 with 0 noise
            camera_noise_end_level = 23        # End at level 23 with max noise
            max_gaussian_noise_std = 0.0125    # Maximum Gaussian noise: 1.25% of depth range
            max_pixel_dropout_rate = 0.025     # Maximum pixel dropout: 2.5% of pixels
            
            # Linear progression from level 3 to 23
            if level <= camera_noise_start_level:
                return 0.0, 0.0  # No noise at level 3
            elif level >= camera_noise_end_level:
                return max_gaussian_noise_std, max_pixel_dropout_rate  # Full noise at level 23
            else:
                # Linear interpolation between start and end levels
                level_progress = (level - camera_noise_start_level) / (camera_noise_end_level - camera_noise_start_level)
                gaussian_std = level_progress * max_gaussian_noise_std
                dropout_rate = level_progress * max_pixel_dropout_rate
                return gaussian_std, dropout_rate

        # REMOVED: get_drone_lateral_offset and get_drone_orientation_randomization
        # These methods have been removed as we now use fixed parameters in LMF2 config
        # with ±0.5m lateral variation and ±45° orientation without curriculum dependency
        
        @staticmethod
        def get_static_camera_difficulty(level):
            """
            Calculate static camera positioning difficulty based on curriculum level.
            
            LINEAR PROGRESSION: Level 3 → Level 23
            - Level 3: 0° max angle range (fixed straight-behind view)
            - Level 23: ±30° max angle range (randomized within full range each episode)
            - Linear interpolation between levels
            
            Returns:
                max_camera_angle: Maximum angle range for randomization (±this value)
                height_offset: Height offset from default position (always 0 - position stays fixed)
                distance_offset: Distance offset from default position (always 0 - position stays fixed)
            """
            # Class constants - LINEAR PROGRESSION
            camera_start_level = 3     # Start progression from level 3
            max_level = 23            # End progression at level 23
            max_camera_angle_degrees = 25  # Maximum ±30° range
            
            # Linear progression from level 3 to 23
            if level <= camera_start_level:
                max_camera_angle = 0.0  # No angle variation at level 3
            elif level >= max_level:
                max_camera_angle = max_camera_angle_degrees  # Full ±30° range at level 23
            else:
                # Linear interpolation between start and end levels
                level_progress = (level - camera_start_level) / (max_level - camera_start_level)
                max_camera_angle = level_progress * max_camera_angle_degrees
            
            # Position stays FIXED - only angle changes
            height_offset = 0.0    # No height variation - keep fixed position
            distance_offset = 0.0  # No distance variation - keep fixed position
            
            return max_camera_angle, height_offset, distance_offset
        
        @staticmethod
        def get_gate_scale_for_level(level):
            """
            Get gate scale factor for current curriculum level with randomization.
            
            PROGRESSIVE GATE SCALING CURRICULUM:
            - Levels 3-8: Only full size gates (scale 1.0) - easiest
            - Levels 9-13: Mix of full and medium gates (scale 1.0, 0.7)
            - Levels 14-18: Mix of full, medium, and small gates (scale 1.0, 0.7, 0.5)
            - Levels 19-23: All gate sizes including minimum (scale 1.0, 0.7, 0.5, 0.4)
            
            Args:
                level: Current curriculum level (3-23)
                
            Returns:
                float: Selected gate scale factor for this episode
            """
            from aerial_gym.config.asset_config.gate_scaling_config import GateScalingConfig
            import random
            
            try:
                # Get available scales for this curriculum level
                available_scales = GateScalingConfig.get_available_scales_for_level(level)
                
                # Debug logging
                print(f"[GATE SCALE SELECT DEBUG] Level {level}: Available scales {available_scales}")
                
                # Randomly select from available scales
                selected_scale = random.choice(available_scales)
                
                print(f"[GATE SCALE SELECT DEBUG] Level {level}: Selected scale {selected_scale}")
                
                return selected_scale
                
            except Exception as e:
                print(f"[GATE SCALE SELECT ERROR] Level {level}: {e}")
                return 1.0  # Fallback to full size
        
        @staticmethod
        def get_gate_tolerance_for_scale(scale_factor):
            """
            Get adaptive success tolerance based on gate scale.
            
            Scales the success tolerance proportionally with gate size:
            - Full size (1.0): ±1.3m width, 0.2-2.2m height (original)
            - Medium size (0.7): ±0.91m width, 0.2-1.6m height
            - Small size (0.5): ±0.65m width, 0.2-1.2m height
            - Minimum size (0.4): ±0.52m width, 0.2-1.0m height
            
            Args:
                scale_factor: Gate scale factor (0.4 to 1.0)
                
            Returns:
                tuple: (width_tolerance, height_min, height_max)
            """
            from aerial_gym.config.asset_config.gate_scaling_config import GateScalingConfig
            return GateScalingConfig.get_gate_tolerance_for_scale(scale_factor)

    # Static camera curriculum positioning based on difficulty level
    class static_camera_curriculum:
        """Progressive static camera positioning to increase difficulty"""
        @staticmethod
        def get_camera_position_and_orientation(level, gate_position, env_bounds):
            """
            Get static camera position and orientation based on curriculum level.
            
            Args:
                level: Current curriculum level (3-23)
                gate_position: Gate position in world coordinates
                env_bounds: Environment bounds [min_x, max_x, min_y, max_y, min_z, max_z]
            
            Returns:
                position: [x, y, z] camera position
                orientation: [x, y, z, w] quaternion orientation (looking at gate)
            """
            # Use the curriculum system to get camera difficulty parameters
            from aerial_gym.config.task_config.navigation_task_config_gate import task_config
            camera_angle, height_offset, distance_offset = task_config.curriculum.get_static_camera_difficulty(level)
            
            # Environment dimensions
            env_width_x = env_bounds[1] - env_bounds[0]  # 8m
            env_width_y = env_bounds[3] - env_bounds[2]  # 8m
            env_height_z = env_bounds[5] - env_bounds[4]  # 4m
            
            # Base camera position (behind gate, looking forward)
            base_distance = 3.0 + distance_offset  # 3.0m to 4.5m behind gate
            base_height = 1.5 + height_offset       # 1.0m to 2.0m height
            
            # Apply angular offset for progressive difficulty
            import math
            angle_rad = math.radians(camera_angle)
            lateral_offset = base_distance * math.sin(angle_rad)
            depth_offset = base_distance * math.cos(angle_rad)
            
            # Position camera with progressive difficulty
            camera_x = gate_position[0] + lateral_offset
            camera_y = gate_position[1] - depth_offset  # Behind gate
            camera_z = base_height
            
            # Ensure camera stays within environment bounds with margin
            margin = 0.5
            camera_x = max(env_bounds[0] + margin, min(env_bounds[1] - margin, camera_x))
            camera_y = max(env_bounds[2] + margin, min(env_bounds[3] - margin, camera_y))
            camera_z = max(env_bounds[4] + margin, min(env_bounds[5] - margin, camera_z))
            
            position = [camera_x, camera_y, camera_z]
            
            # Calculate orientation to look at gate center (always face gate)
            import torch
            camera_pos = torch.tensor(position, dtype=torch.float32)
            gate_pos = torch.tensor(gate_position, dtype=torch.float32)
            
            # Forward direction (camera to gate)
            forward = gate_pos - camera_pos
            forward = forward / torch.norm(forward)
            
            # Up direction
            up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
            
            # Right direction (cross product)
            right = torch.cross(forward, up)
            right = right / torch.norm(right)
            
            # Recalculate up to ensure orthogonality
            up = torch.cross(right, forward)
            
            # Create rotation matrix
            rotation_matrix = torch.stack([right, -forward, up], dim=1)
            
            # Convert to quaternion (simplified - looking along negative Y)
            # For a camera looking in +Y direction (towards gate), we need rotation
            # This is a simplified quaternion for looking towards +Y
            if forward[1] > 0:  # Looking towards positive Y (towards gate)
                # Look at gate center
                orientation = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion
            else:
                # Look towards gate (negative Y to positive Y)
                orientation = [0.0, 0.0, 1.0, 0.0]  # 180° around Z-axis
            
            return position, orientation

    # VELOCITY CONTROLLER: 4D Action transformation for direct velocity control
    @staticmethod
    @torch.jit.script
    def action_transformation_4d_velocity_control(actions):
        # type: (Tensor) -> Tensor
        """
        Transform 4D actions for VELOCITY CONTROLLER - conservative scaling for stability.
        Input: [x_vel_cmd, y_vel_cmd, z_vel_cmd, yaw_rate_cmd] ∈ [-1, 1]^4
        Output: [x_vel, y_vel, z_vel, yaw_rate] in real units
        
        Conservative velocity controller constraints for improved stability:
        - X,Y velocity: ±0.6 m/s (CONSERVATIVE scaling for velocity controller stability)
        - Z velocity: ±0.4 m/s (REDUCED for smooth altitude control without position feedback)
        - Yaw rate: ±28.6°/s (±0.5 rad/s) (CONSERVATIVE for smooth yaw without position correction)
        """
        transformed_actions = torch.zeros_like(actions)
        
        # X,Y velocity: ±0.6 m/s (CONSERVATIVE for velocity controller stability)
        transformed_actions[:, 0] = actions[:, 0] * 0.6  # x_vel
        transformed_actions[:, 1] = actions[:, 1] * 0.6  # y_vel
        
        # Z velocity: ±0.4 m/s (REDUCED for smooth altitude control)
        transformed_actions[:, 2] = actions[:, 2] * 0.4  # z_vel
        
        # Yaw rate: ±28.6°/s (±0.5 rad/s) (CONSERVATIVE for smooth yaw control)
        transformed_actions[:, 3] = actions[:, 3] * 0.5  # yaw_rate (28.6 degrees/sec)
        
        return transformed_actions

    action_transformation_function = action_transformation_4d_velocity_control
