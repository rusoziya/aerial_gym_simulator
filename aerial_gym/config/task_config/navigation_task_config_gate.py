import torch
from aerial_gym import AERIAL_GYM_DIRECTORY


class task_config:
    seed = -1
    sim_name = "base_sim"
    env_name = "gate_env"  # Use gate environment instead of env_with_obstacles
    robot_name = "lmf2"  # Use proven LMF2 robot
    controller_name = "lmf2_position_control"  # Use proven LMF2 controller
    args = {}
    num_envs = 16  # Standard configuration for gate navigation training
    use_warp = True
    headless = False  # Enable visualization to view both cameras (can be overridden by Sample Factory)
    device = "cuda:0"
    
    # Enhanced observation space: 17D basic state + 64D drone VAE + 64D static camera VAE = 145D
    # Both cameras now share the same VAE model to reduce GPU memory usage by ~50%
    observation_space_dim = 17 + 64 + 64  # Enhanced with static camera VAE latents
    privileged_observation_space_dim = 0
    action_space_dim = 4  # UPDATED: 4D action space [x_vel, y_vel, z_vel, yaw_rate] for full gate navigation control
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
    target_max_ratio = [0.8, 0.99, 0.45]  # Y=0.99 -> Y=+3.92 (well beyond obstacles), Z=0.45 -> Z=1.8m (within gate)
    
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
        # BASE NAVIGATION REWARDS (Strong penalties to prevent crashes)
        "pos_reward_magnitude": 5.0,  # Match base navigation
        "pos_reward_exponent": 1.0 / 3.5,
        "very_close_to_goal_reward_magnitude": 5.0,  # Match base navigation
        "very_close_to_goal_reward_exponent": 2.0,
        "getting_closer_reward_multiplier": 10.0,  # Match base navigation
        
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
        
        # GATE-SPECIFIC REWARDS (Additive bonuses on top of base rewards)
        "gate_approach_reward_magnitude": 5.0,  # Reward for approaching the gate
        # "gate_approach_reward_magnitude": 2.0,  # Reward for approaching the gate
        "gate_alignment_reward_magnitude": 2.0,  # Reward for aligning with gate opening
        # "gate_alignment_reward_magnitude": 1.0,  # Reward for aligning with gate opening
        "gate_passage_reward_magnitude": 10.0,  # Large reward for successfully passing through gate
        "gate_center_bonus_magnitude": 5.0,  # Bonus for being centered in gate opening
        # "gate_center_passage_bonus_magnitude": 15.0,  # Large bonus for passing through center of gate
        "gate_center_passage_bonus_magnitude": 50.0,  # Large bonus for passing through center of gate
        # "camera_facing_reward_magnitude": 5.0,  # Enhanced reward for drone camera facing towards gate (from user's previous request)
        "camera_facing_reward_magnitude": 5.0,  # Enhanced reward for drone camera facing towards gate (from user's previous request)

        
        # NEW: Altitude maintenance reward to encourage proper gate-level flying
        "altitude_maintenance_reward_magnitude": 2.0,  # Reward for staying at gate height (1.2-1.8m)
        "altitude_maintenance_reward_exponent": 2.0,   # Exponential reward for being at optimal altitude
        
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

    # Gate navigation curriculum with static camera positioning progression
    class curriculum:
        min_level = 3  # Start with 3 obstacles to match gate environment configuration (was 1)
        max_level = 20  # Progress to more difficult static camera positions
        check_after_log_instances = 1024  # Check curriculum progress more frequently for gate task
        increase_step = 1
        decrease_step = 1
        success_rate_for_increase = 0.75  # Higher success rate needed for gate task
        success_rate_for_decrease = 0.5

        def update_curriculim_level(self, success_rate, current_level):
            if success_rate > self.success_rate_for_increase:
                return min(current_level + self.increase_step, self.max_level)
            elif success_rate < self.success_rate_for_decrease:
                return max(current_level - self.decrease_step, self.min_level)
            return current_level

    # Static camera curriculum positioning based on difficulty level
    class static_camera_curriculum:
        """Progressive static camera positioning to increase difficulty"""
        @staticmethod
        def get_camera_position_and_orientation(level, gate_position, env_bounds):
            """
            Get static camera position and orientation based on curriculum level.
            
            Args:
                level: Current curriculum level (3-20)
                gate_position: Gate position in world coordinates
                env_bounds: Environment bounds [min_x, max_x, min_y, max_y, min_z, max_z]
            
            Returns:
                position: [x, y, z] camera position
                orientation: [x, y, z, w] quaternion orientation (looking at gate)
            """
            # Normalize level to [0, 1] range
            level_progress = (level - 3) / (20 - 3)  # 0.0 at level 3, 1.0 at level 20
            
            # Environment dimensions
            env_width_x = env_bounds[1] - env_bounds[0]  # 8m
            env_width_y = env_bounds[3] - env_bounds[2]  # 8m
            env_height_z = env_bounds[5] - env_bounds[4]  # 4m
            
            # Camera height progression: start low, increase with difficulty
            camera_height = 1.0 + level_progress * 1.5  # 1.0m to 2.5m
            
            # Camera distance from gate: start close, move further with difficulty  
            base_distance = 3.0  # meters behind gate
            distance_variation = level_progress * 1.5  # up to 1.5m further
            camera_distance = base_distance + distance_variation
            
            # Camera lateral offset: start centered, add offset with difficulty
            max_lateral_offset = 1.5  # maximum lateral displacement
            lateral_offset = (level_progress - 0.5) * max_lateral_offset  # -0.75m to +0.75m
            
            # Position camera behind gate (negative Y direction)
            camera_x = gate_position[0] + lateral_offset
            camera_y = gate_position[1] - camera_distance  # Behind gate
            camera_z = camera_height
            
            # Ensure camera stays within environment bounds with margin
            margin = 0.5
            camera_x = max(env_bounds[0] + margin, min(env_bounds[1] - margin, camera_x))
            camera_y = max(env_bounds[2] + margin, min(env_bounds[3] - margin, camera_y))
            camera_z = max(env_bounds[4] + margin, min(env_bounds[5] - margin, camera_z))
            
            position = [camera_x, camera_y, camera_z]
            
            # Calculate orientation to look at gate center
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

    # UPDATED: 4D Action transformation for gate navigation with Z-axis control
    @staticmethod
    @torch.jit.script
    def action_transformation_4d_gate_navigation(actions):
        # type: (Tensor) -> Tensor
        """
        Transform 4D actions with Z-axis control within gate height limits.
        Input: [x_vel_cmd, y_vel_cmd, z_vel_cmd, yaw_rate_cmd] ∈ [-1, 1]^4
        Output: [x_vel, y_vel, z_vel, yaw_rate] in real units
        
        Gate navigation constraints:
        - X,Y velocity: ±1.0 m/s (REDUCED from 2.0 for more precise control)
        - Z velocity: ±1.0 m/s (constrained by gate height 0.2-2.2m)
        - Yaw rate: ±60°/s (±1.047 rad/s)
        """
        transformed_actions = torch.zeros_like(actions)
        
        # X,Y velocity: ±1.0 m/s (REDUCED for more precise gate navigation)
        transformed_actions[:, 0] = actions[:, 0] * 1.0  # x_vel
        transformed_actions[:, 1] = actions[:, 1] * 1.0  # y_vel
        
        # Z velocity: ±1.0 m/s (constrained by gate height 0.2-2.2m)
        transformed_actions[:, 2] = actions[:, 2] * 1.0  # z_vel
        
        # Yaw rate: ±60°/s (±1.047 rad/s)
        transformed_actions[:, 3] = actions[:, 3] * 1.047  # yaw_rate (60 degrees/sec)
        
        return transformed_actions

    action_transformation_function = action_transformation_4d_gate_navigation
