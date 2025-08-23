import torch
from aerial_gym import AERIAL_GYM_DIRECTORY


class task_config:
    seed = -1
    sim_name = "base_sim"
    env_name = "gate_env"  # Use gate environment instead of env_with_obstacles
    robot_name = "lmf2"  # Use proven LMF2 robot
    controller_name = "lmf2_velocity_control"  # CHANGED: Switch to velocity controller for more direct control
    args = {}
    num_envs = 16  # Default to 16 environments for gate navigation training
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
        "z_action_diff_penalty_exponent": 2.0,  # REDUCED from 3.333 for even gentler Z-smoothness

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
        max_pixel_dropout_rate = 0.0125          # Halved: 1.25% of pixels
        
        # 6. CAMERA FRAME DROPOUT (entire-frame) parameters (split freeze/blank)
        enable_camera_frame_dropout = True
        max_frame_freeze_prob_drone = 0.05       # 5% freeze at level 23
        max_frame_blank_prob_drone = 0.005       # 0.5% blank at level 23
        max_frame_freeze_prob_static = 0.05      # 5% freeze at level 23
        max_frame_blank_prob_static = 0.005      # 0.5% blank at level 23
        frame_dropout_start_level = 3
        frame_dropout_end_level = 23
        # Back-compat flag; when split is used, freeze is default when not blank
        frame_dropout_mode = "freeze"
 
        # 7. STATE/POSE NOISE (drone & static) parameters
        enable_state_noise = True
        state_noise_start_level = 3
        state_noise_end_level = 23
        # Maxima at level 23
        max_drone_pos_noise_m = 0.02                # per-axis std in meters
        max_static_pos_noise_m = 0.05               # per-axis std in meters
        max_static_orient_noise_rad = 0.017453292519943295  # 1.0 deg
        max_drone_orient_noise_rad = 0.008726646259971648   # 0.5 deg

        # 8. SPAWN RANGE PROGRESSION (Levels 3-23) — curriculum-controlled spawn
        spawn_start_level = 3
        spawn_end_level = 23
        # Easy (level 3) spawn: tight
        spawn_easy_x_half_span_m = 0.20
        spawn_easy_y_center_m = -2.0
        spawn_easy_y_half_span_m = 0.05
        spawn_easy_z_center_m = 1.5
        spawn_easy_z_half_span_m = 0.05
        # At the easiest level, force camera to face the gate: 0° jitter
        spawn_easy_yaw_abs_rad = 0.0 * 3.141592653589793 / 180.0
        # Hard (level 23) spawn: matches prior LMF2
        spawn_hard_x_half_span_m = 1.20
        spawn_hard_y_center_m = -2.0
        spawn_hard_y_half_span_m = 0.20
        spawn_hard_z_center_m = 1.5
        spawn_hard_z_half_span_m = 0.10
        # Reduce maximum spawn yaw jitter at hardest level to 30°
        spawn_hard_yaw_abs_rad = 30.0 * 3.141592653589793 / 180.0
        
        # EVALUATION PARAMETERS
        check_after_log_instances = 256  # Check curriculum every 256 instances for reduced variance
        increase_step = 1  # Increase by 1 level at a time for fine-grained progression
        decrease_step = 1  # Allow decreases by 1 level when success collapses
        success_rate_for_increase = 0.6  # Promote when SR > 60%
        success_rate_for_decrease = 0.25   # Demote when SR < 25%
        cooldown_windows = 3  # After any change, hold level for this many evaluation windows
        
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
            max_pixel_dropout_rate = 0.0125    # Halved: 1.25% of pixels
            
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

        @staticmethod
        def get_camera_frame_dropout(level):
            """
            Linear schedules for entire-frame dropouts with split freeze/blank probabilities.
            Returns a dict with keys:
              - 'drone_freeze', 'drone_blank', 'static_freeze', 'static_blank'
              - 'drone_total' (freeze+blank), 'static_total' (freeze+blank)
            """
            start = task_config.curriculum.frame_dropout_start_level
            end = task_config.curriculum.frame_dropout_end_level
            if level <= start:
                return {
                    "drone_freeze": 0.0,
                    "drone_blank": 0.0,
                    "static_freeze": 0.0,
                    "static_blank": 0.0,
                    "drone_total": 0.0,
                    "static_total": 0.0,
                }
            if level >= end:
                df = task_config.curriculum.max_frame_freeze_prob_drone
                db = task_config.curriculum.max_frame_blank_prob_drone
                sf = task_config.curriculum.max_frame_freeze_prob_static
                sb = task_config.curriculum.max_frame_blank_prob_static
                return {
                    "drone_freeze": df,
                    "drone_blank": db,
                    "static_freeze": sf,
                    "static_blank": sb,
                    "drone_total": df + db,
                    "static_total": sf + sb,
                }
            # Linear interpolation between start and end levels
            progress = (level - start) / float(end - start)
            df = progress * task_config.curriculum.max_frame_freeze_prob_drone
            db = progress * task_config.curriculum.max_frame_blank_prob_drone
            sf = progress * task_config.curriculum.max_frame_freeze_prob_static
            sb = progress * task_config.curriculum.max_frame_blank_prob_static
            return {
                "drone_freeze": df,
                "drone_blank": db,
                "static_freeze": sf,
                "static_blank": sb,
                "drone_total": df + db,
                "static_total": sf + sb,
            }

        @staticmethod
        def get_state_noise(level):
            """
            Linear schedules for state/pose noise (drone & static), per-axis Gaussian stds.
            Returns dict with keys:
              - drone_pos_std_m, drone_orient_std_rad
              - static_pos_std_m, static_orient_std_rad
            """
            start = task_config.curriculum.state_noise_start_level
            end = task_config.curriculum.state_noise_end_level
            if level <= start:
                return {
                    "drone_pos_std_m": 0.0,
                    "drone_orient_std_rad": 0.0,
                    "static_pos_std_m": 0.0,
                    "static_orient_std_rad": 0.0,
                }
            if level >= end:
                return {
                    "drone_pos_std_m": task_config.curriculum.max_drone_pos_noise_m,
                    "drone_orient_std_rad": task_config.curriculum.max_drone_orient_noise_rad,
                    "static_pos_std_m": task_config.curriculum.max_static_pos_noise_m,
                    "static_orient_std_rad": task_config.curriculum.max_static_orient_noise_rad,
                }
            progress = (level - start) / float(end - start)
            return {
                "drone_pos_std_m": progress * task_config.curriculum.max_drone_pos_noise_m,
                "drone_orient_std_rad": progress * task_config.curriculum.max_drone_orient_noise_rad,
                "static_pos_std_m": progress * task_config.curriculum.max_static_pos_noise_m,
                "static_orient_std_rad": progress * task_config.curriculum.max_static_orient_noise_rad,
            }

        @staticmethod
        def get_spawn_ranges(level):
            """
            Linear spawn-range schedule from level 3 to 23.
            Returns dict with:
              - x_half_span_m
              - y_center_m, y_half_span_m
              - z_center_m, z_half_span_m
              - yaw_abs_rad
            """
            s = task_config.curriculum.spawn_start_level
            e = task_config.curriculum.spawn_end_level
            if level <= s:
                return {
                    "x_half_span_m": task_config.curriculum.spawn_easy_x_half_span_m,
                    "y_center_m": task_config.curriculum.spawn_easy_y_center_m,
                    "y_half_span_m": task_config.curriculum.spawn_easy_y_half_span_m,
                    "z_center_m": task_config.curriculum.spawn_easy_z_center_m,
                    "z_half_span_m": task_config.curriculum.spawn_easy_z_half_span_m,
                    "yaw_abs_rad": task_config.curriculum.spawn_easy_yaw_abs_rad,
                }
            if level >= e:
                return {
                    "x_half_span_m": task_config.curriculum.spawn_hard_x_half_span_m,
                    "y_center_m": task_config.curriculum.spawn_hard_y_center_m,
                    "y_half_span_m": task_config.curriculum.spawn_hard_y_half_span_m,
                    "z_center_m": task_config.curriculum.spawn_hard_z_center_m,
                    "z_half_span_m": task_config.curriculum.spawn_hard_z_half_span_m,
                    "yaw_abs_rad": task_config.curriculum.spawn_hard_yaw_abs_rad,
                }
            p = (level - s) / float(e - s)
            def lerp(a, b):
                return a + p * (b - a)
            return {
                "x_half_span_m": lerp(task_config.curriculum.spawn_easy_x_half_span_m, task_config.curriculum.spawn_hard_x_half_span_m),
                "y_center_m": lerp(task_config.curriculum.spawn_easy_y_center_m, task_config.curriculum.spawn_hard_y_center_m),
                "y_half_span_m": lerp(task_config.curriculum.spawn_easy_y_half_span_m, task_config.curriculum.spawn_hard_y_half_span_m),
                "z_center_m": lerp(task_config.curriculum.spawn_easy_z_center_m, task_config.curriculum.spawn_hard_z_center_m),
                "z_half_span_m": lerp(task_config.curriculum.spawn_easy_z_half_span_m, task_config.curriculum.spawn_hard_z_half_span_m),
                "yaw_abs_rad": lerp(task_config.curriculum.spawn_easy_yaw_abs_rad, task_config.curriculum.spawn_hard_yaw_abs_rad),
            }

        @staticmethod
        def get_static_camera_difficulty(level):
            """
            Calculate static camera positioning difficulty based on curriculum level.
            
            LINEAR PROGRESSION: Level 3 → Level 23
            - Level 3: 0° max angle range (fixed straight-behind view)
            - Level 23: ±19° max angle range (randomized within full range each episode)
            - Linear interpolation between levels
            
            Returns:
                max_camera_angle: Maximum angle range for randomization (±this value)
                height_offset: Height offset from default position (always 0 - position stays fixed)
                distance_offset: Distance offset from default position (always 0 - position stays fixed)
            """
            camera_start_level = 3
            max_level = 23
            max_camera_angle_degrees = 19
            if level <= camera_start_level:
                max_camera_angle = 0.0
            elif level >= max_level:
                max_camera_angle = max_camera_angle_degrees
            else:
                level_progress = (level - camera_start_level) / (max_level - camera_start_level)
                max_camera_angle = level_progress * max_camera_angle_degrees
            height_offset = 0.0
            distance_offset = 0.0
            return max_camera_angle, height_offset, distance_offset

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
