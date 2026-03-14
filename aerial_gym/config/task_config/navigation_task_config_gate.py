import torch
import os
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
    # Minimal static camera placement randomization (non-intrusive defaults)
    static_camera_randomize_placement = True
    static_camera_min_translation = [-0.02, -0.02, -0.01]
    static_camera_max_translation = [0.02, 0.02, 0.01]
    # Euler jitter kept tiny; yaw jitter is auto-disabled when curriculum yaw/sweep active
    static_camera_min_euler_deg = [-1.0, -0.5, 0.0]   # roll, pitch, yaw (deg)
    static_camera_max_euler_deg = [1.0,   0.5, 0.0]
    
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

    # Ablation flags (defaults; overridable via SF_DISABLE_* env vars)
    disable_obstacle_randomization = False
    fixed_obstacles_behind_gate = 0
    disable_gate_size_randomization = False
    fixed_gate_scale_percent = 100
    disable_static_camera_orientation_randomization = False
    disable_camera_noise_randomization = False
    disable_camera_frame_dropout_randomization = False
    disable_spawn_position_randomization = False
    disable_spawn_orientation_randomization = False
    disable_state_noise_randomization = False
    disable_dynamic_camera_following = False
    disable_curriculum_multiplier = False

    # Static camera base placement (overridable via SF_STATIC_CAMERA_BASE_Y/Z env vars)
    static_camera_base_y = -3.0
    static_camera_base_z = 1.5

    # Forced curriculum level override (None = use normal progression)
    force_curriculum_level = None
    max_curriculum_level = None  # None = use curriculum.max_level

    # DEBUG TOGGLES
    guard_debug_enabled = True
    enable_comprehensive_reward_debug = False
    reward_outlier_threshold = -180.0
    reward_outlier_log_limit_per_step = 8
   
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
    
    # Enhanced reward parameters for gate navigation with 4D action space
    reward_parameters = {
        # BASE NAVIGATION REWARDS (SCALED DOWN to prevent reward inflation)
        "pos_reward_magnitude": 0.5,  # Further reduced from 2.5
        "pos_reward_exponent": 0.15,
        "very_close_to_goal_reward_magnitude": 0.75,  # REDUCED from 5.0 (10x reduction)
        "very_close_to_goal_reward_exponent": 1.25,
        "getting_closer_reward_multiplier": 5.0,  # REDUCED from 10.0 (10x reduction)
        
        # Action smoothness penalties (match base navigation)
        # gate config default
        "x_action_diff_penalty_magnitude": 0.8,
        "x_action_diff_penalty_exponent": 3.333,
        "y_action_diff_penalty_magnitude": 0.8,  # Y-axis smoothness penalty  
        "y_action_diff_penalty_exponent": 3.333,

        # navigation original:
        # "z_action_diff_penalty_magnitude": 0.8,  # REDUCED from 0.8 to allow necessary altitude adjustments
        # "z_action_diff_penalty_exponent": 5.0,  # REDUCED from 3.333 for gentler Z-smoothness

        # gate config default
        "z_action_diff_penalty_magnitude": 0.4,  # REDUCED from 0.8 to allow necessary altitude adjustments
        "z_action_diff_penalty_exponent": 2.0,  # REDUCED from 3.333 for even gentler Z-smoothness

        # "yawrate_action_diff_penalty_magnitude": 0.8,
        # "yawrate_action_diff_penalty_exponent": 3.33,
        "yawrate_action_diff_penalty_magnitude": 0.5,  # REDUCED from 0.8 for less restrictive yaw control
        "yawrate_action_diff_penalty_exponent": 2.5,  # REDUCED from 3.33 for less restrictive yaw control
        
        # Action magnitude penalties (match base navigation for crash prevention)#
        # gate config default
        "x_absolute_action_penalty_magnitude": 0.1,
        "x_absolute_action_penalty_exponent": 0.3,
        "y_absolute_action_penalty_magnitude": 0.1,  # Y-axis action penalty
        "y_absolute_action_penalty_exponent": 0.3,

        # navigation original:
        # "z_absolute_action_penalty_magnitude": 1.5,  # Strong vertical movement penalty
        # "z_absolute_action_penalty_exponent": 1.0,
        # "z_absolute_action_penalty_magnitude": 0.1,  # REDUCED from 0.1 to allow necessary Z-movements
        # "z_absolute_action_penalty_exponent": 0.3,   # REDUCED from 0.3 for even gentler Z-penalty

        # gate config default
        "z_absolute_action_penalty_magnitude": 0.05,  # REDUCED from 0.1 to allow necessary Z-movements
        "z_absolute_action_penalty_exponent": 0.2,   # REDUCED from 0.3 for even gentler Z-penalty

        # navigation original:
        # "yawrate_absolute_action_penalty_magnitude": 1.5,
        # "yawrate_absolute_action_penalty_exponent": 2.0,
        
        "yawrate_absolute_action_penalty_magnitude": 1.0,  # REDUCED from 1.5 for less restrictive yaw control
        "yawrate_absolute_action_penalty_exponent": 1.1,  # REDUCED from 2.0 for less restrictive yaw control
        
        # CRITICAL: Strong collision penalty to prevent ground crashes
        "collision_penalty": -100.0,  # Match base navigation - STRONG penalty for any collision
        
        # GATE-SPECIFIC REWARDS (SCALED DOWN to prevent reward inflation)
        "gate_approach_reward_magnitude": 1.25,  # REDUCED from 5.0 (10x reduction)
        # "gate_approach_reward_magnitude": 2.0,  # OLD value
        "gate_alignment_reward_magnitude": 0.5,  # REDUCED from 2.0 (10x reduction)
        # "gate_alignment_reward_magnitude": 1.0,  # OLD value
        "gate_passage_reward_magnitude": 100.0,  # INCREASED from 10.0 (one-time bonus should be high)
        "gate_center_bonus_magnitude": 1.25,  # REDUCED from 5.0 (10x reduction)
        # "gate_center_passage_bonus_magnitude": 15.0,  # OLD value
        "gate_center_passage_bonus_magnitude": 100.0,  # INCREASED from 50.0 (one-time center bonus should be high)
        # "camera_facing_reward_magnitude": 5.0,  # Enhanced reward for drone camera facing towards gate (from user's previous request)
        "camera_facing_reward_magnitude": 0.0,  # Enhanced reward for drone camera facing towards gate (TRIPLED from 1.0)

        # IMAGE-BASED PENALTY (depth min distance, 0.4m-20.0m range)
        # Penalty applied as -exp(-(d^2)*exponent) * magnitude for non-terminated envs
        "image_penalty_magnitude": 0.2,
        "image_penalty_exponent": 2.0,

        # Static camera FOV visibility reward (shaped inside-frustum score)
        # Set magnitude > 0 to enable; exponent controls rolloff near edges
        # "static_fov_visibility_reward_magnitude": 0.2,
        "static_fov_visibility_reward_magnitude": 0.0,
        "static_fov_visibility_exponent": 2.0,

        # NEW: Altitude maintenance reward to encourage proper gate-level flying
        "altitude_maintenance_reward_magnitude": 0.0,  # Reward for staying at gate height (1.2-1.8m)
        "altitude_maintenance_reward_exponent": 0.0,   # Exponential reward for being at optimal altitude
        
        # Gate collision penalty (separate from general collision for specificity)
        "gate_collision_penalty": -50.0,  # Additional penalty for hitting gate specifically
        # Boundary violation penalty magnitude (applied once per episode when crossing to front side outside passage window)
        "boundary_violation_penalty_magnitude": 50.0,

        # TIME-BASED PENALTIES (configurable)
        # Per-step time cost target at horizon (total, before reward_scale): set to 4.0 to yield ≈-4 across 100 steps
        "time_penalty_total_at_horizon": 4.0,
        # Shape parameters for r_time = -lambda0 * (1 + lambda1 * s^p), s = step/horizon
        "time_penalty_lambda1": 1.0,
        "time_penalty_exponent": 2.0,
        # Optional explicit lambda0 override (normally computed from total@horizon); set None to auto-compute
        # "time_penalty_lambda0": 0.0,
        # One-off penalty applied only at timeout (no success/crash)
        "timeout_penalty": 75.0,
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
        
        # EVAL-ONLY SCHEDULE STRETCHING (does not affect training)
        # When enabled, curriculum schedules linearly interpolate to this end level during evaluation
        # so that values that normally reach their maximum at level=23 instead reach the same maximum at level=eval_stretch_end_level.
        # Example: with eval_stretch_enabled=True and eval_stretch_end_level=30, progress is computed over [3..30].
        eval_stretch_enabled = bool(int(os.getenv('EVAL_STRETCH_ENABLED', '0')))
        eval_stretch_end_level = int(os.getenv('EVAL_STRETCH_END_LEVEL', '33'))
        # Optional: allow obstacles to increase beyond level 23 during eval stretch
        stretched_end_obstacles = int(os.getenv('EVAL_STRETCH_END_OBSTACLES', '25'))
        
        # ===== 5. CAMERA NOISE PROGRESSION (Levels 3-23) - D455 Realistic Noise =====
        # Simulate Intel RealSense D455 camera characteristics for both drone and static cameras
        enable_camera_noise = True                # Enable curriculum-dependent camera noise
        camera_noise_start_level = 3             # Start minimal noise from level 3 
        camera_noise_end_level = 23              # Reach maximum noise at level 23
        max_gaussian_noise_std = 0.00625         # Halved: 0.625% of depth range
        max_pixel_dropout_rate = 0.00625         # Halved: 0.625% of pixels

        # max_gaussian_noise_std = 0.0125          # Maximum Gaussian noise: 1.25% of depth range
        # max_pixel_dropout_rate = 0.0125          # Halved: 1.25% of pixels

        # 6. CAMERA FRAME DROPOUT (entire-frame) parameters (split freeze/blank)
        enable_camera_frame_dropout = True
        max_frame_freeze_prob_drone = 0.0125      # Halved: 1.25% freeze at level 23
        max_frame_blank_prob_drone = 0.0025      # Halved: 0.25% blank at level 23
        max_frame_freeze_prob_static = 0.0125     # Halved: 1.25% freeze at level 23
        max_frame_blank_prob_static = 0.0025     # Halved: 0.25% blank at level 23
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
        max_static_orient_noise_rad = 0.0175  # 1.0 deg
        max_drone_orient_noise_rad = 0.00873   # 0.5 deg

        # 8. SPAWN RANGE PROGRESSION (Levels 3-23) — curriculum-controlled spawn
        # OLD (kept for easy rollback):
        # spawn_easy_x_half_span_m = 0.20
        # spawn_easy_y_center_m = -2.0
        # spawn_easy_y_half_span_m = 0.05
        # spawn_easy_z_center_m = 1.5
        # spawn_easy_z_half_span_m = 0.05
        # spawn_hard_x_half_span_m = 1.20
        # spawn_hard_y_center_m = -2.0
        # spawn_hard_y_half_span_m = 0.20
        # spawn_hard_z_center_m = 1.5
        # spawn_hard_z_half_span_m = 0.10
        
        spawn_start_level = 3
        spawn_end_level = 23
        
        # NEW (request):
        # - X half-span: ±0.5m at L3 → ±2.0m at L23 (linear)
        # - Y fixed at -1.5m (no randomization)
        # - Z range: [0.75, 1.50] at L3 → [0.50, 1.75] at L23 (linear)
        #   Both intervals are centered at 1.125m, so center stays constant; half-span grows 0.375 → 0.625
        spawn_easy_x_half_span_m = 0.50
        spawn_hard_x_half_span_m = 1.50
        
        spawn_easy_y_center_m = -1.50
        spawn_hard_y_center_m = -1.50
        # Remove Y randomization by setting half-span to 0 at all levels
        spawn_easy_y_half_span_m = 0.00
        spawn_hard_y_half_span_m = 0.00
        
        spawn_easy_z_center_m = 1.125
        spawn_hard_z_center_m = 1.125
        spawn_easy_z_half_span_m = 0.375  # 1.125±0.375 ⇒ [0.75, 1.50]
        spawn_hard_z_half_span_m = 0.625  # 1.125±0.625 ⇒ [0.50, 1.75]
        
        # Yaw jitter schedule: keep 0° at easy; 30° at hard (unchanged)
        spawn_easy_yaw_abs_rad = 1.0 * 3.141592653589793 / 180.0
        spawn_hard_yaw_abs_rad = 20.0 * 3.141592653589793 / 180.0
        
        # EVALUATION PARAMETERS
        check_after_log_instances = 256  # Check curriculum every 256 instances for reduced variance
        increase_step = 1  # Increase by 1 level at a time for fine-grained progression
        decrease_step = 1  # Allow decreases by 1 level when success collapses
        # success_rate_for_increase = 0.5  # Promote when SR > 50%
        # success_rate_for_decrease = 0.25   # Demote when SR < 25%
        success_rate_for_increase = 0.55  # Promote when SR > 55%
        success_rate_for_decrease = 0.25   # Demote when SR < 30%
        cooldown_windows = 12  # After any change, hold level for this many evaluation windows
        avg3_success_for_increase = None  # Optional rolling-3 success threshold (None = disabled)
        
        # MULTI-ASPECT DIFFICULTY PROGRESSION
        # Each curriculum level controls multiple aspects of difficulty:
        
        # 1. OBSTACLE COUNT PROGRESSION (Levels 3-23)
        # Linear schedule: Level 3 → 3 obstacles, Level 23 → 10 obstacles
        # Intermediate levels interpolate linearly and are rounded to nearest integer
        max_obstacles_behind_gate = 10  # Final cap at level 23
        
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
        
        # DYNAMIC CAMERA FOLLOWING CONFIGURATION
        # Toggle between static camera (curriculum-based positioning) and dynamic following
        enable_dynamic_camera_following = False  # Toggle: False = static camera, True = follow drone
        
        # Dynamic camera offset relative to drone position (in meters)
        # Camera position = drone_position + offset_vector
        # Camera always looks towards ADAPTIVE gate center (varies per environment based on gate size/position)
        dynamic_camera_follow_distance_x = 0.0   # No lateral offset - directly behind drone
        dynamic_camera_follow_distance_y = -1.0  # Distance behind drone in Y direction (meters)
        dynamic_camera_follow_distance_z = 0.0   # No Z offset as requested (camera at same height as drone)
        
        # DEBUGGING AND MONITORING
        enable_detailed_logging = True  # Enable comprehensive curriculum debugging
        log_curriculum_changes = True   # Log all curriculum aspect changes
        save_curriculum_metrics = True  # Save curriculum metrics to wandb
        
        # CAMERA BEHAVIOR LOGGING
        log_camera_following = False    # Log dynamic camera following behavior (can be verbose)

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
            Number of obstacles behind gate based on curriculum level (3→23).
            - Level 3: 3 obstacles
            - Level 23: 10 obstacles
            - Linear interpolation in between (rounded to nearest int)
            """
            min_level = task_config.curriculum.min_level
            max_level = task_config.curriculum.max_level
            # Use stretched end level for evaluation if enabled
            effective_max_level = (
                task_config.curriculum.eval_stretch_end_level
                if task_config.curriculum.eval_stretch_enabled
                else max_level
            )
            start_obstacles = 3
            end_obstacles = 10
            stretched_end_obstacles = task_config.curriculum.stretched_end_obstacles
            total_asset_capacity = 30  # Must match gate_object_params.num_assets in gate_env.py
            
            # Piecewise linear progression:
            # - Level 3..23: 3 -> 10
            # - Level 23..effective_max_level (when eval stretch enabled): 10 -> stretched_end_obstacles
            lvl = max(min_level, level)
            if lvl <= max_level:
                progress = (lvl - min_level) / float(max_level - min_level) if max_level > min_level else 1.0
                requested_obstacles = int(round(start_obstacles + progress * (end_obstacles - start_obstacles)))
            else:
                # Evaluation stretch: extrapolate linearly using the same slope as training (3..23)
                upper = max(max_level, effective_max_level)
                lvl_clamped = min(lvl, upper)
                span_train = float(max(1, max_level - min_level))
                slope = (end_obstacles - start_obstacles) / span_train  # obstacles per level in training
                extra_levels = float(lvl_clamped - max_level)
                requested_obstacles = int(round(end_obstacles + slope * extra_levels))
            
            # Safety clamps
            if requested_obstacles > total_asset_capacity:
                print(f"WARNING: Curriculum requested {requested_obstacles} obstacles but only {total_asset_capacity} available!")
                requested_obstacles = total_asset_capacity
            return requested_obstacles
        
        @staticmethod
        def get_camera_noise(level):
            """
            Calculate camera noise parameters based on curriculum level.
            
            LINEAR PROGRESSION: Level 3 → Level 23
            - Level 3: Non-zero starting values (5% of max)
            - Level 23: 0.0125 Gaussian noise, 0.0125 dropout (maximum values)
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
            camera_noise_start_level = 3       # Start at level 3
            camera_noise_end_train = 23        # End of training schedule
            # Level 3 starting values (5% of max) and Level 23 maximum values (training caps)
            max_gaussian_noise_std = 0.00625   # Level 23: 0.00625 (halved)
            max_pixel_dropout_rate = 0.00625   # Level 23: 0.00625 (halved)
            min_gaussian_noise_std = max_gaussian_noise_std * 0.05    # Level 3: 0.000625 (5% of max)
            min_pixel_dropout_rate = max_pixel_dropout_rate * 0.05    # Level 3: 0.000625 (5% of max)
            
            # Compute slope per level for training range 3..23
            span_train = float(max(1, camera_noise_end_train - camera_noise_start_level))
            slope_gauss = (max_gaussian_noise_std - min_gaussian_noise_std) / span_train
            slope_drop = (max_pixel_dropout_rate - min_pixel_dropout_rate) / span_train

            lvl = max(camera_noise_start_level, level)
            if lvl <= camera_noise_end_train:
                progress = (lvl - camera_noise_start_level) / span_train
                gaussian_std = min_gaussian_noise_std + progress * (max_gaussian_noise_std - min_gaussian_noise_std)
                dropout_rate = min_pixel_dropout_rate + progress * (max_pixel_dropout_rate - min_pixel_dropout_rate)
            else:
                # Evaluation stretch: extrapolate beyond training cap using the same slope
                eval_end = int(task_config.curriculum.eval_stretch_end_level)
                if not task_config.curriculum.eval_stretch_enabled:
                    eval_end = camera_noise_end_train
                lvl_clamped = min(lvl, eval_end)
                extra = float(lvl_clamped - camera_noise_end_train)
                gaussian_std = max_gaussian_noise_std + slope_gauss * extra
                dropout_rate = max_pixel_dropout_rate + slope_drop * extra
            return gaussian_std, dropout_rate

        @staticmethod
        def get_camera_frame_dropout(level):
            """
            Linear schedules for entire-frame dropouts with split freeze/blank probabilities.
            
            LINEAR PROGRESSION: Level 3 → Level 23
            - Level 3: Non-zero starting values (5% of max)
            - Level 23: 5.0% freeze, 0.5% blank (maximum values)
            - Linear interpolation between levels
            
            Returns a dict with keys:
              - 'drone_freeze', 'drone_blank', 'static_freeze', 'static_blank'
              - 'drone_total' (freeze+blank), 'static_total' (freeze+blank)
            """
            start = task_config.curriculum.frame_dropout_start_level  # Level 3
            end_train = task_config.curriculum.frame_dropout_end_level  # Level 23
            
            # Define start and end values for linear interpolation
            max_drone_freeze = task_config.curriculum.max_frame_freeze_prob_drone  # Level 23: 5%
            max_drone_blank = task_config.curriculum.max_frame_blank_prob_drone    # Level 23: 0.5%
            max_static_freeze = task_config.curriculum.max_frame_freeze_prob_static # Level 23: 5%
            max_static_blank = task_config.curriculum.max_frame_blank_prob_static   # Level 23: 0.5%
            
            # Level 3 starting values (5% of max)
            min_drone_freeze = max_drone_freeze * 0.05    # Level 3: 0.25% (5% of 5%)
            min_drone_blank = max_drone_blank * 0.05      # Level 3: 0.025% (5% of 0.5%)
            min_static_freeze = max_static_freeze * 0.05  # Level 3: 0.25% (5% of 5%)
            min_static_blank = max_static_blank * 0.05    # Level 3: 0.025% (5% of 0.5%)
            
            lvl = max(start, level)
            span_train = float(max(1, end_train - start))
            slope_df = (max_drone_freeze - min_drone_freeze) / span_train
            slope_db = (max_drone_blank - min_drone_blank) / span_train
            slope_sf = (max_static_freeze - min_static_freeze) / span_train
            slope_sb = (max_static_blank - min_static_blank) / span_train

            if lvl <= end_train:
                progress = (lvl - start) / span_train
                df = min_drone_freeze + progress * (max_drone_freeze - min_drone_freeze)
                db = min_drone_blank + progress * (max_drone_blank - min_drone_blank)
                sf = min_static_freeze + progress * (max_static_freeze - min_static_freeze)
                sb = min_static_blank + progress * (max_static_blank - min_static_blank)
            else:
                eval_end = int(task_config.curriculum.eval_stretch_end_level)
                if not task_config.curriculum.eval_stretch_enabled:
                    eval_end = end_train
                lvl_clamped = min(lvl, eval_end)
                extra = float(lvl_clamped - end_train)
                df = max_drone_freeze + slope_df * extra
                db = max_drone_blank + slope_db * extra
                sf = max_static_freeze + slope_sf * extra
                sb = max_static_blank + slope_sb * extra
            
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
            
            LINEAR PROGRESSION: Level 3 → Level 23
            - Level 3: Non-zero starting values (5% of max)
            - Level 23: Maximum values (drone pos 0.02m, drone orient 0.5°, static pos 0.05m, static orient 1.0°)
            - Linear interpolation between levels
            
            Returns dict with keys:
              - drone_pos_std_m, drone_orient_std_rad
              - static_pos_std_m, static_orient_std_rad
            """
            start = task_config.curriculum.state_noise_start_level  # Level 3
            end_train = task_config.curriculum.state_noise_end_level  # Level 23
            
            # Define start and end values for linear interpolation
            max_drone_pos_noise = task_config.curriculum.max_drone_pos_noise_m  # Level 23: 0.02m
            max_drone_orient_noise = task_config.curriculum.max_drone_orient_noise_rad  # Level 23: 0.5°
            max_static_pos_noise = task_config.curriculum.max_static_pos_noise_m  # Level 23: 0.05m
            max_static_orient_noise = task_config.curriculum.max_static_orient_noise_rad  # Level 23: 1.0°
            
            # Level 3 starting values (5% of max)
            min_drone_pos_noise = max_drone_pos_noise * 0.05  # Level 3: 0.001m (5% of 0.02m)
            min_drone_orient_noise = max_drone_orient_noise * 0.05  # Level 3: ~0.025° (5% of 0.5°)
            min_static_pos_noise = max_static_pos_noise * 0.05  # Level 3: 0.0025m (5% of 0.05m)
            min_static_orient_noise = max_static_orient_noise * 0.05  # Level 3: ~0.05° (5% of 1.0°)
            
            lvl = max(start, level)
            span_train = float(max(1, end_train - start))
            slopes = {
                "drone_pos_std_m": (max_drone_pos_noise - min_drone_pos_noise) / span_train,
                "drone_orient_std_rad": (max_drone_orient_noise - min_drone_orient_noise) / span_train,
                "static_pos_std_m": (max_static_pos_noise - min_static_pos_noise) / span_train,
                "static_orient_std_rad": (max_static_orient_noise - min_static_orient_noise) / span_train,
            }
            if lvl <= end_train:
                progress = (lvl - start) / span_train
                return {
                    "drone_pos_std_m": min_drone_pos_noise + progress * (max_drone_pos_noise - min_drone_pos_noise),
                    "drone_orient_std_rad": min_drone_orient_noise + progress * (max_drone_orient_noise - min_drone_orient_noise),
                    "static_pos_std_m": min_static_pos_noise + progress * (max_static_pos_noise - min_static_pos_noise),
                    "static_orient_std_rad": min_static_orient_noise + progress * (max_static_orient_noise - min_static_orient_noise),
                }
            else:
                eval_end = int(task_config.curriculum.eval_stretch_end_level)
                if not task_config.curriculum.eval_stretch_enabled:
                    eval_end = end_train
                lvl_clamped = min(lvl, eval_end)
                extra = float(lvl_clamped - end_train)
                return {
                    "drone_pos_std_m": max_drone_pos_noise + slopes["drone_pos_std_m"] * extra,
                    "drone_orient_std_rad": max_drone_orient_noise + slopes["drone_orient_std_rad"] * extra,
                    "static_pos_std_m": max_static_pos_noise + slopes["static_pos_std_m"] * extra,
                    "static_orient_std_rad": max_static_orient_noise + slopes["static_orient_std_rad"] * extra,
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
            e_train = task_config.curriculum.spawn_end_level  # 23
            if level <= s:
                return {
                    "x_half_span_m": task_config.curriculum.spawn_easy_x_half_span_m,
                    "y_center_m": task_config.curriculum.spawn_easy_y_center_m,
                    "y_half_span_m": task_config.curriculum.spawn_easy_y_half_span_m,
                    "z_center_m": task_config.curriculum.spawn_easy_z_center_m,
                    "z_half_span_m": task_config.curriculum.spawn_easy_z_half_span_m,
                    "yaw_abs_rad": task_config.curriculum.spawn_easy_yaw_abs_rad,
                }
            if level >= e_train:
                if not task_config.curriculum.eval_stretch_enabled:
                    return {
                        "x_half_span_m": task_config.curriculum.spawn_hard_x_half_span_m,
                        "y_center_m": task_config.curriculum.spawn_hard_y_center_m,
                        "y_half_span_m": task_config.curriculum.spawn_hard_y_half_span_m,
                        "z_center_m": task_config.curriculum.spawn_hard_z_center_m,
                        "z_half_span_m": task_config.curriculum.spawn_hard_z_half_span_m,
                        "yaw_abs_rad": task_config.curriculum.spawn_hard_yaw_abs_rad,
                    }
                # Evaluation stretch: extrapolate beyond hard values using training slope
                span_train = float(max(1, e_train - s))
                def lerp(a, b):
                    return a + (e_train - s) / span_train * (b - a)
                slopes = {
                    "x_half_span_m": (task_config.curriculum.spawn_hard_x_half_span_m - task_config.curriculum.spawn_easy_x_half_span_m) / span_train,
                    "y_center_m": (task_config.curriculum.spawn_hard_y_center_m - task_config.curriculum.spawn_easy_y_center_m) / span_train,
                    "y_half_span_m": (task_config.curriculum.spawn_hard_y_half_span_m - task_config.curriculum.spawn_easy_y_half_span_m) / span_train,
                    "z_center_m": (task_config.curriculum.spawn_hard_z_center_m - task_config.curriculum.spawn_easy_z_center_m) / span_train,
                    "z_half_span_m": (task_config.curriculum.spawn_hard_z_half_span_m - task_config.curriculum.spawn_easy_z_half_span_m) / span_train,
                    "yaw_abs_rad": (task_config.curriculum.spawn_hard_yaw_abs_rad - task_config.curriculum.spawn_easy_yaw_abs_rad) / span_train,
                }
                eval_end = int(task_config.curriculum.eval_stretch_end_level)
                lvl_clamped = min(level, eval_end)
                extra = float(lvl_clamped - e_train)
                return {
                    "x_half_span_m": task_config.curriculum.spawn_hard_x_half_span_m + slopes["x_half_span_m"] * extra,
                    "y_center_m": task_config.curriculum.spawn_hard_y_center_m + slopes["y_center_m"] * extra,
                    "y_half_span_m": task_config.curriculum.spawn_hard_y_half_span_m + slopes["y_half_span_m"] * extra,
                    "z_center_m": task_config.curriculum.spawn_hard_z_center_m + slopes["z_center_m"] * extra,
                    "z_half_span_m": task_config.curriculum.spawn_hard_z_half_span_m + slopes["z_half_span_m"] * extra,
                    "yaw_abs_rad": task_config.curriculum.spawn_hard_yaw_abs_rad + slopes["yaw_abs_rad"] * extra,
                }
            p = (level - s) / float(e_train - s)
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
            # End at level 23 in training; optionally stretch to eval_stretch_end_level during evaluation
            max_level = (
                task_config.curriculum.eval_stretch_end_level
                if task_config.curriculum.eval_stretch_enabled
                else 23
            )
            max_camera_angle_degrees = 19
            min_camera_angle_degrees = 2.0  # NEW: ensure ±2° minimum at level 3
            if level <= camera_start_level:
                max_camera_angle = min_camera_angle_degrees
            elif level >= max_level:
                max_camera_angle = max_camera_angle_degrees
            else:
                level_progress = (level - camera_start_level) / (max_level - camera_start_level)
                max_camera_angle = min_camera_angle_degrees + level_progress * (max_camera_angle_degrees - min_camera_angle_degrees)
            height_offset = 0.0
            distance_offset = 0.0
            return max_camera_angle, height_offset, distance_offset
        
        @staticmethod
        def get_dynamic_camera_follow_offset():
            """
            Get the offset vector for dynamic camera following.
            
            Returns:
                tuple: (x_offset, y_offset, z_offset) in meters
            """
            return (
                task_config.curriculum.dynamic_camera_follow_distance_x,
                task_config.curriculum.dynamic_camera_follow_distance_y, 
                task_config.curriculum.dynamic_camera_follow_distance_z
            )

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
        Transform 4D actions for VELOCITY CONTROLLER with conservative default limits.
        Input: [x_vel_cmd, y_vel_cmd, z_vel_cmd, yaw_rate_cmd] ∈ [-1, 1]^4
        Output: [x_vel, y_vel, z_vel, yaw_rate] in real units
        
        Default limits (reverted, +25% yaw):
        - X velocity: ±0.6 m/s
        - Y velocity: ±0.6 m/s
        - Z velocity: ±0.4 m/s
        - Yaw rate: ±0.625 rad/s (≈ ±35.8°/s)
        """
        transformed_actions = torch.zeros_like(actions)
        
        # X, Y velocities
        transformed_actions[:, 0] = actions[:, 0] * 0.6  # x_vel
        transformed_actions[:, 1] = actions[:, 1] * 0.6  # y_vel
        
        # Z velocity
        transformed_actions[:, 2] = actions[:, 2] * 0.4  # z_vel
        
        # Yaw rate
        transformed_actions[:, 3] = actions[:, 3] * 0.5  # yaw_rate


        return transformed_actions

    action_transformation_function = action_transformation_4d_velocity_control
