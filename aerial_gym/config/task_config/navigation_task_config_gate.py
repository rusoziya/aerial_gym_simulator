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
    action_space_dim = 3  # Keep 3D action space for compatibility with existing inference
    episode_len_steps = 300  # Longer episodes needed for challenging spawn positions and navigation

    return_state_before_reset = (
        False  # False as usually state is returned for next episode after reset
    )
    # user can set the above to true if they so desire

    # Target positions (goals) - keep targets on front side of gate
    # Front side (positive Y) where obstacles are, forcing gate navigation
    # Obstacles at Y = [+2.0, +3.2], so targets should be beyond Y = +3.6
    target_min_ratio = [0.2, 0.95, 0.3]  # Y=0.95 -> Y=+3.6 (beyond obstacles)
    target_max_ratio = [0.8, 0.99, 0.7]  # Y=0.99 -> Y=+3.92 (well beyond obstacles)
    
    # GATE ORIENTATION ANALYSIS (CRITICAL) - UPDATED FOR 8x8x4 ENVIRONMENT:
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
    # Gate-specific reward parameters - STRONG base rewards + gate-specific bonuses
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
        "z_action_diff_penalty_magnitude": 0.8,  # Strong vertical smoothness penalty
        "z_action_diff_penalty_exponent": 5.0,
        "yawrate_action_diff_penalty_magnitude": 0.8,
        "yawrate_action_diff_penalty_exponent": 3.33,
        
        # Action magnitude penalties (match base navigation for crash prevention)
        "x_absolute_action_penalty_magnitude": 0.1,
        "x_absolute_action_penalty_exponent": 0.3,
        "z_absolute_action_penalty_magnitude": 1.5,  # Strong vertical movement penalty
        "z_absolute_action_penalty_exponent": 1.0,
        "yawrate_absolute_action_penalty_magnitude": 1.5,
        "yawrate_absolute_action_penalty_exponent": 2.0,
        
        # CRITICAL: Strong collision penalty to prevent ground crashes
        "collision_penalty": -100.0,  # Match base navigation - STRONG penalty for any collision
        
        # GATE-SPECIFIC REWARDS (Additive bonuses on top of base rewards)
        "gate_approach_reward_magnitude": 2.0,  # Reward for approaching the gate
        "gate_alignment_reward_magnitude": 1.0,  # Reward for aligning with gate opening
        "gate_passage_reward_magnitude": 10.0,  # Large reward for successfully passing through gate
        "gate_center_bonus_magnitude": 3.0,  # Bonus for being centered in gate opening
        "gate_center_passage_bonus_magnitude": 15.0,  # Large bonus for passing through center of gate
        "camera_facing_reward_magnitude": 3.0,  # NEW: Reward for drone camera facing towards gate
        
        # Gate collision penalty (separate from general collision for specificity)
        "gate_collision_penalty": -20.0,  # Additional penalty for hitting gate specifically
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
        @staticmethod
        def get_camera_position_and_orientation(level, gate_position, env_bounds):
            """
            Get static camera position and orientation based on curriculum level.
            
            Level 1-5: Camera directly behind gate, perfect view
            Level 6-10: Camera slightly offset, still good view
            Level 11-15: Camera more offset, angled view
            Level 16-20: Camera at challenging angles, partial view
            """
            level = max(1, min(20, level))  # Clamp to valid range
            
            # Gate is at center (0,0,0), environment bounds are [-4,4] x [-4,4] x [0,4]
            gate_x, gate_y, gate_z = gate_position
            
            if level <= 5:
                # Easy: Camera directly behind gate, perfect alignment
                camera_x = gate_x
                camera_y = gate_y - 3.0  # 3.0m behind gate (scaled for smaller environment)
                camera_z = gate_z + 1.0  # 1.0m above ground
                # Look directly at gate center
                target_x, target_y, target_z = gate_x, gate_y, gate_z + 0.8
                
            elif level <= 10:
                # Medium: Camera slightly offset horizontally
                offset_factor = (level - 5) / 5.0  # 0.0 to 1.0
                camera_x = gate_x + offset_factor * 1.0  # Offset up to 1m (scaled for smaller env)
                camera_y = gate_y - 3.0
                camera_z = gate_z + 1.0
                # Still look at gate center
                target_x, target_y, target_z = gate_x, gate_y, gate_z + 0.8
                
            elif level <= 15:
                # Medium: Side angle and varying height
                angle_factor = (level - 10) / 5.0  # 0.0 to 1.0
                camera_x = gate_x + 2.0 * angle_factor  # Side position up to 2m (scaled for smaller env)
                camera_y = gate_y - 2.0  # 2m behind gate (scaled for smaller env)
                camera_z = gate_z + 1.0 + angle_factor * 1.0  # Height varies from 1m to 2m
                # Look at gate center
                target_x, target_y, target_z = gate_position
            else:
                # Hard: Extreme angles and positions (level 16-20)
                extreme_factor = (level - 15) / 5.0  # 0.0 to 1.0
                camera_x = gate_x + 3.0 * extreme_factor * (1 if level % 2 == 0 else -1)  # Alternate sides
                camera_y = gate_y - 2.5 - extreme_factor * 1.0  # Further back, up to 3.5m
                camera_z = gate_z + 2.0 + extreme_factor * 1.0  # Higher camera, up to 3m
                # Look at gate center
                target_x, target_y, target_z = gate_position
            
            # Ensure camera stays within environment bounds
            env_min_x, env_min_y, env_min_z = env_bounds[0]  # [-4, -4, 0]
            env_max_x, env_max_y, env_max_z = env_bounds[1]  # [4, 4, 4]
            
            camera_x = max(env_min_x + 0.5, min(env_max_x - 0.5, camera_x))
            camera_y = max(env_min_y + 0.5, min(env_max_y - 0.5, camera_y))
            camera_z = max(env_min_z + 0.5, min(env_max_z - 0.5, camera_z))
            
            return (camera_x, camera_y, camera_z), (target_x, target_y, target_z)

    # Gate-specific action transformation function for LMF2
    @staticmethod
    def action_transformation_function(action):
        """
        Transform actions for LMF2 robot with gate navigation.
        Input: 3D action [forward_inclination, lateral_inclination, yaw_rate]
        Output: 4D velocity command [vx, vy, vz, yaw_rate]
        Based on proven LMF2 DCE navigation transformation.
        """
        clamped_action = torch.clamp(action, -1.0, 1.0)
        max_speed = 2.0  # [m/s] - reasonable speed for gate navigation
        max_yaw_rate = torch.pi / 3  # [rad/s] - 60 degrees per second
        max_inclination_angle = torch.pi / 4  # [rad] - 45 degree max inclination

        # Use forward bias for gate navigation (add 1.0 to forward action)
        clamped_action[:, 0] += 1.0

        processed_action = torch.zeros(
            (clamped_action.shape[0], 4), device=clamped_action.device, requires_grad=False
        )
        
        # LMF2-style inclination-based control for gate navigation
        processed_action[:, 0] = (
            clamped_action[:, 0]
            * torch.cos(max_inclination_angle * clamped_action[:, 1])
            * max_speed
            / 2.0
        )
        processed_action[:, 1] = 0.0  # No Y velocity for LMF2
        processed_action[:, 2] = (
            clamped_action[:, 0]
            * torch.sin(max_inclination_angle * clamped_action[:, 1])
            * max_speed
            / 2.0
        )
        processed_action[:, 3] = clamped_action[:, 2] * max_yaw_rate
        
        return processed_action
