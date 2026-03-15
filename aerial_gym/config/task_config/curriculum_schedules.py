from __future__ import annotations


# Lazy import to avoid circular dependency
def _get_curriculum_config():
    from aerial_gym.config.task_config.navigation_task_config_gate import task_config

    return task_config.curriculum


def update_curriculim_level(success_rate, current_level) -> None:
    """
    ENHANCED CURRICULUM UPDATE WITH NO-DECREASE POLICY

    This function implements a no-decrease policy where once a level is reached,
    the difficulty never goes back down. This ensures consistent progression
    and prevents oscillation between difficulty levels.
    """
    # ONLY ALLOW INCREASES (No-decrease policy)
    if success_rate > _get_curriculum_config().success_rate_for_increase:
        new_level = min(
            current_level + _get_curriculum_config().increase_step,
            _get_curriculum_config().max_level,
        )
        return new_level
    else:
        # Maintain current level (never decrease)
        return current_level


def get_obstacle_count_behind_gate(level) -> None:
    """
    Number of obstacles behind gate based on curriculum level (3→23).
    - Level 3: 3 obstacles
    - Level 23: 10 obstacles
    - Linear interpolation in between (rounded to nearest int)
    """
    min_level = _get_curriculum_config().min_level
    max_level = _get_curriculum_config().max_level
    # Use stretched end level for evaluation if enabled
    effective_max_level = (
        _get_curriculum_config().eval_stretch_end_level
        if _get_curriculum_config().eval_stretch_enabled
        else max_level
    )
    start_obstacles = 3
    end_obstacles = 10
    stretched_end_obstacles = _get_curriculum_config().stretched_end_obstacles
    total_asset_capacity = 30  # Must match gate_object_params.num_assets in gate_env.py

    # Piecewise linear progression:
    # - Level 3..23: 3 -> 10
    # - Level 23..effective_max_level (when eval stretch enabled): 10 -> stretched_end_obstacles
    lvl = max(min_level, level)
    if lvl <= max_level:
        progress = (
            (lvl - min_level) / float(max_level - min_level) if max_level > min_level else 1.0
        )
        requested_obstacles = int(
            round(start_obstacles + progress * (end_obstacles - start_obstacles))
        )
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
        print(
            f"WARNING: Curriculum requested {requested_obstacles} obstacles but only {total_asset_capacity} available!"
        )
        requested_obstacles = total_asset_capacity
    return requested_obstacles


def get_camera_noise(level) -> None:
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
    camera_noise_start_level = 3  # Start at level 3
    camera_noise_end_train = 23  # End of training schedule
    # Level 3 starting values (5% of max) and Level 23 maximum values (training caps)
    max_gaussian_noise_std = 0.00625  # Level 23: 0.00625 (halved)
    max_pixel_dropout_rate = 0.00625  # Level 23: 0.00625 (halved)
    min_gaussian_noise_std = max_gaussian_noise_std * 0.05  # Level 3: 0.000625 (5% of max)
    min_pixel_dropout_rate = max_pixel_dropout_rate * 0.05  # Level 3: 0.000625 (5% of max)

    # Compute slope per level for training range 3..23
    span_train = float(max(1, camera_noise_end_train - camera_noise_start_level))
    slope_gauss = (max_gaussian_noise_std - min_gaussian_noise_std) / span_train
    slope_drop = (max_pixel_dropout_rate - min_pixel_dropout_rate) / span_train

    lvl = max(camera_noise_start_level, level)
    if lvl <= camera_noise_end_train:
        progress = (lvl - camera_noise_start_level) / span_train
        gaussian_std = min_gaussian_noise_std + progress * (
            max_gaussian_noise_std - min_gaussian_noise_std
        )
        dropout_rate = min_pixel_dropout_rate + progress * (
            max_pixel_dropout_rate - min_pixel_dropout_rate
        )
    else:
        # Evaluation stretch: extrapolate beyond training cap using the same slope
        eval_end = int(_get_curriculum_config().eval_stretch_end_level)
        if not _get_curriculum_config().eval_stretch_enabled:
            eval_end = camera_noise_end_train
        lvl_clamped = min(lvl, eval_end)
        extra = float(lvl_clamped - camera_noise_end_train)
        gaussian_std = max_gaussian_noise_std + slope_gauss * extra
        dropout_rate = max_pixel_dropout_rate + slope_drop * extra
    return gaussian_std, dropout_rate


def get_camera_frame_dropout(level) -> None:
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
    start = _get_curriculum_config().frame_dropout_start_level  # Level 3
    end_train = _get_curriculum_config().frame_dropout_end_level  # Level 23

    # Define start and end values for linear interpolation
    max_drone_freeze = _get_curriculum_config().max_frame_freeze_prob_drone  # Level 23: 5%
    max_drone_blank = _get_curriculum_config().max_frame_blank_prob_drone  # Level 23: 0.5%
    max_static_freeze = _get_curriculum_config().max_frame_freeze_prob_static  # Level 23: 5%
    max_static_blank = _get_curriculum_config().max_frame_blank_prob_static  # Level 23: 0.5%

    # Level 3 starting values (5% of max)
    min_drone_freeze = max_drone_freeze * 0.05  # Level 3: 0.25% (5% of 5%)
    min_drone_blank = max_drone_blank * 0.05  # Level 3: 0.025% (5% of 0.5%)
    min_static_freeze = max_static_freeze * 0.05  # Level 3: 0.25% (5% of 5%)
    min_static_blank = max_static_blank * 0.05  # Level 3: 0.025% (5% of 0.5%)

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
        eval_end = int(_get_curriculum_config().eval_stretch_end_level)
        if not _get_curriculum_config().eval_stretch_enabled:
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


def get_state_noise(level) -> None:
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
    start = _get_curriculum_config().state_noise_start_level  # Level 3
    end_train = _get_curriculum_config().state_noise_end_level  # Level 23

    # Define start and end values for linear interpolation
    max_drone_pos_noise = _get_curriculum_config().max_drone_pos_noise_m  # Level 23: 0.02m
    max_drone_orient_noise = _get_curriculum_config().max_drone_orient_noise_rad  # Level 23: 0.5°
    max_static_pos_noise = _get_curriculum_config().max_static_pos_noise_m  # Level 23: 0.05m
    max_static_orient_noise = _get_curriculum_config().max_static_orient_noise_rad  # Level 23: 1.0°

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
            "drone_pos_std_m": min_drone_pos_noise
            + progress * (max_drone_pos_noise - min_drone_pos_noise),
            "drone_orient_std_rad": min_drone_orient_noise
            + progress * (max_drone_orient_noise - min_drone_orient_noise),
            "static_pos_std_m": min_static_pos_noise
            + progress * (max_static_pos_noise - min_static_pos_noise),
            "static_orient_std_rad": min_static_orient_noise
            + progress * (max_static_orient_noise - min_static_orient_noise),
        }
    else:
        eval_end = int(_get_curriculum_config().eval_stretch_end_level)
        if not _get_curriculum_config().eval_stretch_enabled:
            eval_end = end_train
        lvl_clamped = min(lvl, eval_end)
        extra = float(lvl_clamped - end_train)
        return {
            "drone_pos_std_m": max_drone_pos_noise + slopes["drone_pos_std_m"] * extra,
            "drone_orient_std_rad": max_drone_orient_noise + slopes["drone_orient_std_rad"] * extra,
            "static_pos_std_m": max_static_pos_noise + slopes["static_pos_std_m"] * extra,
            "static_orient_std_rad": max_static_orient_noise
            + slopes["static_orient_std_rad"] * extra,
        }


def get_spawn_ranges(level) -> None:
    """
    Linear spawn-range schedule from level 3 to 23.
    Returns dict with:
      - x_half_span_m
      - y_center_m, y_half_span_m
      - z_center_m, z_half_span_m
      - yaw_abs_rad
    """
    s = _get_curriculum_config().spawn_start_level
    e_train = _get_curriculum_config().spawn_end_level  # 23
    if level <= s:
        return {
            "x_half_span_m": _get_curriculum_config().spawn_easy_x_half_span_m,
            "y_center_m": _get_curriculum_config().spawn_easy_y_center_m,
            "y_half_span_m": _get_curriculum_config().spawn_easy_y_half_span_m,
            "z_center_m": _get_curriculum_config().spawn_easy_z_center_m,
            "z_half_span_m": _get_curriculum_config().spawn_easy_z_half_span_m,
            "yaw_abs_rad": _get_curriculum_config().spawn_easy_yaw_abs_rad,
        }
    if level >= e_train:
        if not _get_curriculum_config().eval_stretch_enabled:
            return {
                "x_half_span_m": _get_curriculum_config().spawn_hard_x_half_span_m,
                "y_center_m": _get_curriculum_config().spawn_hard_y_center_m,
                "y_half_span_m": _get_curriculum_config().spawn_hard_y_half_span_m,
                "z_center_m": _get_curriculum_config().spawn_hard_z_center_m,
                "z_half_span_m": _get_curriculum_config().spawn_hard_z_half_span_m,
                "yaw_abs_rad": _get_curriculum_config().spawn_hard_yaw_abs_rad,
            }
        # Evaluation stretch: extrapolate beyond hard values using training slope
        span_train = float(max(1, e_train - s))

        def lerp(a, b) -> None:
            return a + (e_train - s) / span_train * (b - a)

        slopes = {
            "x_half_span_m": (
                _get_curriculum_config().spawn_hard_x_half_span_m
                - _get_curriculum_config().spawn_easy_x_half_span_m
            )
            / span_train,
            "y_center_m": (
                _get_curriculum_config().spawn_hard_y_center_m
                - _get_curriculum_config().spawn_easy_y_center_m
            )
            / span_train,
            "y_half_span_m": (
                _get_curriculum_config().spawn_hard_y_half_span_m
                - _get_curriculum_config().spawn_easy_y_half_span_m
            )
            / span_train,
            "z_center_m": (
                _get_curriculum_config().spawn_hard_z_center_m
                - _get_curriculum_config().spawn_easy_z_center_m
            )
            / span_train,
            "z_half_span_m": (
                _get_curriculum_config().spawn_hard_z_half_span_m
                - _get_curriculum_config().spawn_easy_z_half_span_m
            )
            / span_train,
            "yaw_abs_rad": (
                _get_curriculum_config().spawn_hard_yaw_abs_rad
                - _get_curriculum_config().spawn_easy_yaw_abs_rad
            )
            / span_train,
        }
        eval_end = int(_get_curriculum_config().eval_stretch_end_level)
        lvl_clamped = min(level, eval_end)
        extra = float(lvl_clamped - e_train)
        return {
            "x_half_span_m": _get_curriculum_config().spawn_hard_x_half_span_m
            + slopes["x_half_span_m"] * extra,
            "y_center_m": _get_curriculum_config().spawn_hard_y_center_m
            + slopes["y_center_m"] * extra,
            "y_half_span_m": _get_curriculum_config().spawn_hard_y_half_span_m
            + slopes["y_half_span_m"] * extra,
            "z_center_m": _get_curriculum_config().spawn_hard_z_center_m
            + slopes["z_center_m"] * extra,
            "z_half_span_m": _get_curriculum_config().spawn_hard_z_half_span_m
            + slopes["z_half_span_m"] * extra,
            "yaw_abs_rad": _get_curriculum_config().spawn_hard_yaw_abs_rad
            + slopes["yaw_abs_rad"] * extra,
        }
    p = (level - s) / float(e_train - s)

    def lerp(a, b) -> None:
        return a + p * (b - a)

    return {
        "x_half_span_m": lerp(
            _get_curriculum_config().spawn_easy_x_half_span_m,
            _get_curriculum_config().spawn_hard_x_half_span_m,
        ),
        "y_center_m": lerp(
            _get_curriculum_config().spawn_easy_y_center_m,
            _get_curriculum_config().spawn_hard_y_center_m,
        ),
        "y_half_span_m": lerp(
            _get_curriculum_config().spawn_easy_y_half_span_m,
            _get_curriculum_config().spawn_hard_y_half_span_m,
        ),
        "z_center_m": lerp(
            _get_curriculum_config().spawn_easy_z_center_m,
            _get_curriculum_config().spawn_hard_z_center_m,
        ),
        "z_half_span_m": lerp(
            _get_curriculum_config().spawn_easy_z_half_span_m,
            _get_curriculum_config().spawn_hard_z_half_span_m,
        ),
        "yaw_abs_rad": lerp(
            _get_curriculum_config().spawn_easy_yaw_abs_rad,
            _get_curriculum_config().spawn_hard_yaw_abs_rad,
        ),
    }


def get_static_camera_difficulty(level) -> None:
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
        _get_curriculum_config().eval_stretch_end_level
        if _get_curriculum_config().eval_stretch_enabled
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
        max_camera_angle = min_camera_angle_degrees + level_progress * (
            max_camera_angle_degrees - min_camera_angle_degrees
        )
    height_offset = 0.0
    distance_offset = 0.0
    return max_camera_angle, height_offset, distance_offset


def get_dynamic_camera_follow_offset() -> None:
    """
    Get the offset vector for dynamic camera following.

    Returns:
        tuple: (x_offset, y_offset, z_offset) in meters
    """
    return (
        _get_curriculum_config().dynamic_camera_follow_distance_x,
        _get_curriculum_config().dynamic_camera_follow_distance_y,
        _get_curriculum_config().dynamic_camera_follow_distance_z,
    )
